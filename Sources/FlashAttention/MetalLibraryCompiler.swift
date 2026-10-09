//
//  MetalLibraryCompiler.swift
//  FlashAttention
//
//  Transparent CLI fallback for runtime Metal shader compilation.
//

import Foundation
import Metal

/// Compiles Metal shader source to an `MTLLibrary`, transparently falling
/// back to the offline `xcrun metal` toolchain when the runtime JIT rejects
/// inline `__asm` directives.
///
/// On macOS 26 (Tahoe) the runtime compiler (`device.makeLibrary(source:)`)
/// rejects any source containing `__asm(...)` with
/// "illegal string literal in 'asm'". The flash-attention codegen emits
/// `__asm` for simdgroup async-copy intrinsics (`createMetalSimdgroupEvent`),
/// so runtime compilation fails on macOS 26.
///
/// The offline `xcrun metal` compiler still accepts `__asm`, so this helper:
///  1. Tries the runtime JIT first (fast path; works on macOS 15 and earlier,
///     and for kernels that contain no `__asm`).
///  2. On failure, writes the source to a temporary `.metal` file, compiles
///     to `.air` via `xcrun metal`, links to `.metallib` via `xcrun metallib`,
///     then loads the result with `device.makeLibrary(URL:)`.
///
/// Reference: https://github.com/imperatormk/metal24-asm-repro
public enum MetalLibraryCompiler {

  /// Probe source: a declaration using inline `__asm` (an Air simd-group op).
  /// Rejected by the runtime JIT on macOS 26+, accepted by the offline compiler.
  private static let asmProbeSource = """
  #include <metal_stdlib>
  using namespace metal;
  struct _simdgroup_event_t;
  thread _simdgroup_event_t* probe_decl() __asm("air.wait_simdgroup_events");
  kernel void probe_kernel(device uint* out [[buffer(0)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid == 0) { out[0] = 1u; }
  }
  """

  private nonisolated(unsafe) static var asmSupportCache: Bool?
  private nonisolated(unsafe) static let asmSupportLock = NSLock()

  /// Whether inline `__asm` kernels can be compiled in this process: either the
  /// runtime JIT accepts them, or an offline `metal` compiler is available
  /// (Xcode toolchain or Metal Developer Tools via MFA_METAL_TOOLCHAIN).
  /// Cached per process.
  public static func inlineAsmSupported(device: MTLDevice) -> Bool {
    asmSupportLock.lock()
    defer { asmSupportLock.unlock() }
    if let supported = asmSupportCache {
      return supported
    }
    var supported = (try? device.makeLibrary(source: asmProbeSource, options: nil)) != nil
    if !supported, locateToolchain() != nil {
      supported = (try? makeLibraryViaCLI(device: device, source: asmProbeSource, options: nil)) != nil
    }
    asmSupportCache = supported
    return supported
  }

  /// Compiles Metal shader source to an `MTLLibrary`, trying the runtime JIT first and
  /// transparently falling back to the `xcrun metal` CLI when the source
  /// contains `__asm` and the JIT rejects it.
  public static func makeLibrary(
    device: MTLDevice,
    source: String,
    options: MTLCompileOptions? = nil
  ) throws -> MTLLibrary {
    if ProcessInfo.processInfo.environment["MFA_DUMP_SOURCE"] != nil {
      let dumpPath = ProcessInfo.processInfo.environment["MFA_DUMP_SOURCE"]!
      try? source.write(to: URL(fileURLWithPath: dumpPath), atomically: true, encoding: .utf8)
    }
    // MFA_TENSOR=1 forces Metal 4.0 compilation, enabling __HAVE_TENSOR__
    // and the M5 Neural Accelerator tensor path.
    var effectiveOptions = options
    if ProcessInfo.processInfo.environment["MFA_TENSOR"] == "1" {
      let opts = MTLCompileOptions()
      opts.languageVersion = MTLLanguageVersion(rawValue: 1048576)!  // metal4_0
      effectiveOptions = opts
    }
    // MFA_PREFER_CLI=1 forces the offline `xcrun metal` toolchain, which
    // produces measurably faster kernels than the runtime JIT (about 5–10%
    // on large attention shapes). Falls back to the JIT when no toolchain
    // is installed.
    if ProcessInfo.processInfo.environment["MFA_PREFER_CLI"] == "1" {
      if let lib = try? makeLibraryViaCLI(
        device: device, source: source, options: options)
      {
        return lib
      }
    }
    do {
      return try device.makeLibrary(source: source, options: effectiveOptions)
    } catch {
      // macOS 26 (Tahoe) rejects inline __asm in the runtime JIT with
      // "illegal string literal in 'asm'". If the source contains __asm,
      // fall back to the offline CLI compiler which still accepts it.
      // For sources without __asm, rethrow the original JIT error.
      guard source.contains("__asm") else { throw error }
      return try makeLibraryViaCLI(
        device: device, source: source, options: effectiveOptions)
    }
  }

  /// Compiles Metal source via the offline `xcrun metal` toolchain.
  ///
  /// Writes the source to a temp `.metal` file, compiles to `.air`, links to
  /// `.metallib`, then loads via `device.makeLibrary(URL:)`. The language
  /// version from `options` is mapped to the CLI `-std=` flag so that
  /// bfloat / language-version requirements are preserved.
  public static func makeLibraryViaCLI(
    device: MTLDevice,
    source: String,
    options: MTLCompileOptions? = nil
  ) throws -> MTLLibrary {
    let fm = FileManager.default
    let tmp = fm.temporaryDirectory
    let id = UUID().uuidString

    let srcURL = tmp.appendingPathComponent("mfa_\(id).metal")
    let airURL = tmp.appendingPathComponent("mfa_\(id).air")
    let libURL = tmp.appendingPathComponent("mfa_\(id).metallib")

    defer {
      try? fm.removeItem(at: srcURL)
      try? fm.removeItem(at: airURL)
      try? fm.removeItem(at: libURL)
    }

    try source.write(to: srcURL, atomically: true, encoding: .utf8)

    let stdFlag = cliLanguageVersionFlag(options: options)
    let extraFlags = ProcessInfo.processInfo.environment["MFA_METAL_FLAGS"] ?? ""

    guard let toolchain = locateToolchain() else {
      throw NSError(
        domain: "MetalLibraryCompiler", code: 3,
        userInfo: [
          NSLocalizedDescriptionKey: """
          No offline Metal compiler found, so kernels that require inline \
          `__asm` (Air simd-group async copies) cannot be compiled. To enable \
          the asynchronous-copy path, either:
            1. Install Xcode (provides `xcrun metal`), or
            2. Download Apple's Metal Developer Tools and set MFA_METAL_TOOLCHAIN \
          to the directory containing the `metal` and `metallib` executables.
          Pass extra compiler flags through MFA_METAL_FLAGS if needed.
          """
        ])
    }

    // Compile source → .air
    let compile = Self.runShell(
      "\(toolchain.metal) \(stdFlag) \(extraFlags) -c '\(srcURL.path)' -o '\(airURL.path)' 2>&1"
    )
    guard compile.status == 0 else {
      throw NSError(
        domain: "MetalLibraryCompiler", code: 1,
        userInfo: [
          NSLocalizedDescriptionKey: "metal compile failed:\n\(compile.output)"
        ])
    }

    // Link .air → .metallib
    let link = Self.runShell(
      "\(toolchain.metallib) '\(airURL.path)' -o '\(libURL.path)' 2>&1"
    )
    guard link.status == 0 else {
      throw NSError(
        domain: "MetalLibraryCompiler", code: 2,
        userInfo: [
          NSLocalizedDescriptionKey: "metallib link failed:\n\(link.output)"
        ])
    }

    return try device.makeLibrary(URL: libURL)
  }

  /// Locates the offline Metal compiler. Checks, in order:
  /// 1. `MFA_METAL_TOOLCHAIN` — a directory containing `metal` and `metallib`
  ///    executables (e.g. from Apple's Metal Developer Tools package).
  /// 2. `xcrun` — present when Xcode with the Metal toolchain is installed.
  private static func locateToolchain() -> (metal: String, metallib: String)? {
    let fm = FileManager.default
    let environment = ProcessInfo.processInfo.environment

    if let dir = environment["MFA_METAL_TOOLCHAIN"], !dir.isEmpty {
      for sub in ["", "bin"] {
        let metal = (dir as NSString).appendingPathComponent(sub).appending("/metal")
        let metallib = (dir as NSString).appendingPathComponent(sub).appending("/metallib")
        if fm.isExecutableFile(atPath: metal), fm.isExecutableFile(atPath: metallib) {
          return ("'\(metal)'", "'\(metallib)'")
        }
      }
    }

    let xcrunMetal = runShell("xcrun -f -sdk macosx metal 2>/dev/null")
    let xcrunMetallib = runShell("xcrun -f -sdk macosx metallib 2>/dev/null")
    if xcrunMetal.status == 0, xcrunMetallib.status == 0 {
      let metal = xcrunMetal.output.trimmingCharacters(in: .whitespacesAndNewlines)
      let metallib = xcrunMetallib.output.trimmingCharacters(in: .whitespacesAndNewlines)
      if fm.isExecutableFile(atPath: metal), fm.isExecutableFile(atPath: metallib) {
        return ("'\(metal)'", "'\(metallib)'")
      }
    }

    return nil
  }

  /// Maps an `MTLCompileOptions.languageVersion` to the corresponding
  /// `-std=macos-metalX.Y` CLI flag. Defaults to `macos-metal3.2` when
  /// no language version is specified, matching `mfaCompileOptions()`.
  private static func cliLanguageVersionFlag(
    options: MTLCompileOptions?
  ) -> String {
    // MFA_TENSOR=1 forces Metal 4.0, enabling __HAVE_TENSOR__
    if ProcessInfo.processInfo.environment["MFA_TENSOR"] == "1" {
      return "-std=metal4.0"
    }
    let version: MTLLanguageVersion = options?.languageVersion ?? .version3_2
    switch version {
    case .version3_2: return "-std=metal3.2"
    case .version3_1: return "-std=metal3.1"
    case .version3_0: return "-std=metal3.0"
    case .version2_4: return "-std=macos-metal2.4"
    case .version2_3: return "-std=macos-metal2.3"
    case .version2_2: return "-std=macos-metal2.2"
    case .version2_1: return "-std=macos-metal2.1"
    case .version2_0: return "-std=macos-metal2.0"
    case .version1_2: return "-std=macos-metal1.2"
    case .version1_1: return "-std=macos-metal1.1"
    @unknown default: return "-std=metal3.2"
    }
  }

  private static func runShell(
    _ command: String
  ) -> (output: String, status: Int32) {
    let task = Process()
    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = pipe
    task.launchPath = "/bin/bash"
    task.arguments = ["-c", command]
    do {
      try task.run()
      task.waitUntilExit()
    } catch {
      return ("Failed to launch: \(error)", -1)
    }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    return (String(data: data, encoding: .utf8) ?? "", task.terminationStatus)
  }
}
