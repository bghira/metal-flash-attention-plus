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

  /// Compiles Metal source to a library, trying the runtime JIT first and
  /// transparently falling back to the `xcrun metal` CLI when the source
  /// contains `__asm` and the JIT rejects it.
  public static func makeLibrary(
    device: MTLDevice,
    source: String,
    options: MTLCompileOptions? = nil
  ) throws -> MTLLibrary {
    do {
      return try device.makeLibrary(source: source, options: options)
    } catch {
      // macOS 26 (Tahoe) rejects inline __asm in the runtime JIT with
      // "illegal string literal in 'asm'". If the source contains __asm,
      // fall back to the offline CLI compiler which still accepts it.
      // For sources without __asm, rethrow the original JIT error.
      guard source.contains("__asm") else { throw error }
      return try makeLibraryViaCLI(
        device: device, source: source, options: options)
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

    // Compile source → .air
    let compile = Self.runShell(
      "xcrun -sdk macosx metal \(stdFlag) -c '\(srcURL.path)' -o '\(airURL.path)' 2>&1"
    )
    guard compile.status == 0 else {
      throw NSError(
        domain: "MetalLibraryCompiler", code: 1,
        userInfo: [
          NSLocalizedDescriptionKey: "xcrun metal compile failed:\n\(compile.output)"
        ])
    }

    // Link .air → .metallib
    let link = Self.runShell(
      "xcrun -sdk macosx metallib '\(airURL.path)' -o '\(libURL.path)' 2>&1"
    )
    guard link.status == 0 else {
      throw NSError(
        domain: "MetalLibraryCompiler", code: 2,
        userInfo: [
          NSLocalizedDescriptionKey: "xcrun metallib link failed:\n\(link.output)"
        ])
    }

    return try device.makeLibrary(URL: libURL)
  }

  /// Maps an `MTLCompileOptions.languageVersion` to the corresponding
  /// `-std=macos-metalX.Y` CLI flag. Defaults to `macos-metal3.2` when
  /// no language version is specified, matching `mfaCompileOptions()`.
  private static func cliLanguageVersionFlag(
    options: MTLCompileOptions?
  ) -> String {
    let version: MTLLanguageVersion = options?.languageVersion ?? .version3_2
    switch version {
    case .version3_2: return "-std=macos-metal3.2"
    case .version3_1: return "-std=macos-metal3.1"
    case .version3_0: return "-std=macos-metal3.0"
    case .version2_4: return "-std=macos-metal2.4"
    case .version2_3: return "-std=macos-metal2.3"
    case .version2_2: return "-std=macos-metal2.2"
    case .version2_1: return "-std=macos-metal2.1"
    case .version2_0: return "-std=macos-metal2.0"
    case .version1_2: return "-std=macos-metal1.2"
    case .version1_1: return "-std=macos-metal1.1"
    @unknown default: return "-std=macos-metal3.2"
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
