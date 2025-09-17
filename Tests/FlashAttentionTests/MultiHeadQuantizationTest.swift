//
//  MultiHeadQuantizationTest.swift
//
//  Tests for multi-head attention with quantization strategy fields
//

import XCTest
import Metal
@testable import FlashAttention

final class MultiHeadQuantizationTest: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var mha: MultiHeadAttention!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device should be available")
        commandQueue = device.makeCommandQueue()
        XCTAssertNotNil(commandQueue, "Command queue should be created")
        mha = MultiHeadAttention(device: device)
    }

    func testMultiHeadQuantizationStrategyEncoding() throws {
        // Test dimensions
        let batchSize: UInt32 = 2
        let numHeads: UInt32 = 8
        let numKVHeads: UInt32 = 4  // Grouped query attention
        let seqLength: UInt32 = 64
        let headDim: UInt16 = 32

        // Create test buffers
        let qSize = Int(batchSize * numHeads * seqLength) * Int(headDim) * 2  // FP16
        let kvSize = Int(batchSize * numKVHeads * seqLength) * Int(headDim) * 2  // FP16

        let queryBuffer = device.makeBuffer(length: qSize, options: .storageModeShared)!
        let keyBuffer = device.makeBuffer(length: kvSize, options: .storageModeShared)!
        let valueBuffer = device.makeBuffer(length: kvSize, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: qSize, options: .storageModeShared)!

        // Initialize buffers with test data
        let qPtr = queryBuffer.contents().bindMemory(to: Float16.self, capacity: qSize/2)
        let kPtr = keyBuffer.contents().bindMemory(to: Float16.self, capacity: kvSize/2)
        let vPtr = valueBuffer.contents().bindMemory(to: Float16.self, capacity: kvSize/2)

        for i in 0..<(qSize/2) { qPtr[i] = Float16(1.0) }
        for i in 0..<(kvSize/2) { kPtr[i] = Float16(0.5) }
        for i in 0..<(kvSize/2) { vPtr[i] = Float16(0.25) }

        // Create attention descriptor with symmetric quantization
        var baseDesc = AttentionDescriptor()
        baseDesc.lowPrecisionInputs = false
        baseDesc.lowPrecisionIntermediates = false
        baseDesc.matrixDimensions = (row: seqLength, column: seqLength, head: headDim)
        baseDesc.transposeState = (Q: false, K: false, V: false, O: false)
        baseDesc.sparsityPattern = .none
        baseDesc.softmaxScale = 1.0 / sqrtf(Float(headDim))

        // Create quantization parameters with symmetric strategy
        let symmetricParams = QuantizationParameters(
            scale: 0.0625,  // 1/16
            zeroPoint: 0,   // Symmetric has zero point of 0
            precision: .INT8,
            mode: .tensorWise,
            strategy: .symmetric,
            strategyVersion: 1
        )

        let asymmetricParams = QuantizationParameters(
            scale: 0.0625,
            zeroPoint: 128,  // Asymmetric with non-zero point
            precision: .INT8,
            mode: .tensorWise,
            strategy: .asymmetric,
            strategyVersion: 1
        )

        // Test different quantization configurations
        let testConfigs: [(String, [AttentionOperand: QuantizationParameters])] = [
            ("All Symmetric", [
                .Q: symmetricParams,
                .K: symmetricParams,
                .V: symmetricParams,
                .O: symmetricParams
            ]),
            ("Mixed Strategy", [
                .Q: asymmetricParams,
                .K: symmetricParams,
                .V: symmetricParams,
                .O: asymmetricParams
            ]),
            ("Partial Quantization", [
                .Q: symmetricParams,
                .K: asymmetricParams
                // V and O remain unquantized
            ])
        ]

        for (testName, quantParams) in testConfigs {
            let descriptor = MultiHeadAttentionDescriptor(
                baseDescriptor: baseDesc,
                queryShape: MultiHeadShape(
                    batchSize: batchSize,
                    numHeads: numHeads,
                    sequenceLength: seqLength,
                    headDimension: headDim
                ),
                keyShape: MultiHeadShape(
                    batchSize: batchSize,
                    numHeads: numKVHeads,
                    sequenceLength: seqLength,
                    headDimension: headDim
                ),
                valueShape: MultiHeadShape(
                    batchSize: batchSize,
                    numHeads: numKVHeads,
                    sequenceLength: seqLength,
                    headDimension: headDim
                ),
                broadcastMode: .groupedQuery(numKVHeads: numKVHeads),
                dispatchStrategy: .perBatchHead,
                quantizationParameters: quantParams
            )

            // Verify quantization bindings are created correctly
            let bindings = mha.quantizationBindings(for: descriptor)
            XCTAssertEqual(bindings.count, quantParams.count,
                          "\(testName): Should have \(quantParams.count) quantization bindings")

            // Verify binding order and parameters
            for binding in bindings {
                guard let expectedParams = quantParams[binding.operand] else {
                    XCTFail("\(testName): Unexpected binding for operand \(binding.operand)")
                    continue
                }

                XCTAssertEqual(binding.parameters.scale, expectedParams.scale,
                              "\(testName): Scale mismatch for \(binding.operand)")
                XCTAssertEqual(binding.parameters.zeroPoint, expectedParams.zeroPoint,
                              "\(testName): Zero point mismatch for \(binding.operand)")
                XCTAssertEqual(binding.parameters.strategy, expectedParams.strategy,
                              "\(testName): Strategy mismatch for \(binding.operand)")
                XCTAssertEqual(binding.parameters.strategyVersion, expectedParams.strategyVersion,
                              "\(testName): Strategy version mismatch for \(binding.operand)")
            }

            // Execute attention to ensure encoding works
            let commandBuffer = mha.forward(
                query: queryBuffer,
                key: keyBuffer,
                value: valueBuffer,
                output: outputBuffer,
                descriptor: descriptor
            )

            XCTAssertNotNil(commandBuffer, "\(testName): Command buffer should be created")
            commandBuffer?.commit()
            commandBuffer?.waitUntilCompleted()

            // Verify output is non-zero
            let outPtr = outputBuffer.contents().bindMemory(to: Float16.self, capacity: qSize/2)
            var hasNonZero = false
            for i in 0..<min(100, qSize/2) {
                if outPtr[i] != 0 {
                    hasNonZero = true
                    break
                }
            }
            XCTAssertTrue(hasNonZero, "\(testName): Output should contain non-zero values")
        }
    }

    func testMultiHeadQuantizationWithDifferentDispatchStrategies() throws {
        let batchSize: UInt32 = 1
        let numHeads: UInt32 = 4
        let seqLength: UInt32 = 32
        let headDim: UInt16 = 16

        let bufferSize = Int(batchSize * numHeads * seqLength) * Int(headDim) * 2
        let queryBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        let keyBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        let valueBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!

        // Initialize with test pattern
        let qPtr = queryBuffer.contents().bindMemory(to: Float16.self, capacity: bufferSize/2)
        for i in 0..<(bufferSize/2) { qPtr[i] = Float16(1.0 / Float(i + 1)) }

        let strategies: [MultiHeadDispatchStrategy] = [
            .perBatchHead,
            .perBatch,
            .batched
        ]

        let quantParams: [AttentionOperand: QuantizationParameters] = [
            .Q: QuantizationParameters(scale: 0.125, zeroPoint: 0, precision: .INT8,
                                     mode: .tensorWise, strategy: .symmetric, strategyVersion: 1),
            .K: QuantizationParameters(scale: 0.25, zeroPoint: 64, precision: .INT8,
                                     mode: .tensorWise, strategy: .asymmetric, strategyVersion: 1)
        ]

        for strategy in strategies {
            var baseDesc = AttentionDescriptor()
            baseDesc.lowPrecisionInputs = false
            baseDesc.lowPrecisionIntermediates = false
            baseDesc.matrixDimensions = (row: seqLength, column: seqLength, head: headDim)
            baseDesc.transposeState = (Q: false, K: false, V: false, O: false)
            baseDesc.sparsityPattern = .none
            baseDesc.softmaxScale = 1.0 / sqrtf(Float(headDim))

            let shape = MultiHeadShape(
                batchSize: batchSize,
                numHeads: numHeads,
                sequenceLength: seqLength,
                headDimension: headDim
            )

            let descriptor = MultiHeadAttentionDescriptor(
                baseDescriptor: baseDesc,
                queryShape: shape,
                keyShape: shape,
                valueShape: shape,
                broadcastMode: .standard,
                dispatchStrategy: strategy,
                quantizationParameters: quantParams
            )

            let commandBuffer = mha.forward(
                query: queryBuffer,
                key: keyBuffer,
                value: valueBuffer,
                output: outputBuffer,
                descriptor: descriptor
            )

            XCTAssertNotNil(commandBuffer, "Strategy \(strategy): Should create command buffer")
            commandBuffer?.commit()
            commandBuffer?.waitUntilCompleted()

            if let error = commandBuffer?.error {
                XCTFail("Strategy \(strategy): Command buffer error: \(error)")
            }
        }
    }
}