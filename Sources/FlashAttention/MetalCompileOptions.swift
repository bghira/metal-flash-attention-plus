import Metal

public extension MTLCompileOptions {
  /// Compile options that force Metal Shading Language 3.2.
  ///
  /// Native bfloat16 (`__HAVE_BFLOAT__`) is gated on the MSL version passed to
  /// `device.makeLibrary`. Some macOS 15 runtimes default to a lower language
  /// version for runtime-compiled source than the offline `metal` compiler
  /// uses, which made generated bfloat kernels fail with
  /// "unknown type name 'bfloat'". Forcing `.version3_2` makes runtime
  /// compilation match the offline behaviour so native bfloat is available
  /// consistently.
  static var mfaDefault: MTLCompileOptions {
    let options = MTLCompileOptions()
    options.languageVersion = .version3_2
    return options
  }
}
