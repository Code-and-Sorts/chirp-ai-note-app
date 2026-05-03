import AVFoundation
import CoreMedia
import Darwin
import Foundation
import ScreenCaptureKit

let stderrHandle = FileHandle.standardError
let stdoutHandle = FileHandle.standardOutput

var startupFailed = false

func writeStderr(_ line: String) {
    if let data = line.data(using: .utf8) {
        do {
            try stderrHandle.write(contentsOf: data)
        } catch {
            handleBrokenPipe()
        }
    }
}

func flushStderr() {
    try? stderrHandle.synchronize()
}

func handleBrokenPipe() {
    captureSession?.shutdown(reason: "epipe")
    flushStderr()
    exit(startupFailed ? 1 : 0)
}

func failStartup(_ message: String) -> Never {
    startupFailed = true
    writeStderr("error: \(message)\n")
    flushStderr()
    exit(1)
}

final class CaptureSession: NSObject, SCStreamOutput, SCStreamDelegate {
    private let writeQueue = DispatchQueue(label: "com.codeandsorts.chirp.capture-audio.write")
    private let anchorHostTime: UInt64
    private let timebaseNumer: UInt64
    private let timebaseDenom: UInt64
    private var stream: SCStream?
    private let audioEngine = AVAudioEngine()
    private var stoppedLock = os_unfair_lock()
    private var stopped = false
    private var micConverterErrorReported = false

    override init() {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        self.timebaseNumer = UInt64(info.numer)
        self.timebaseDenom = UInt64(info.denom)
        self.anchorHostTime = mach_absolute_time()
        super.init()
    }

    private func isStopped() -> Bool {
        os_unfair_lock_lock(&stoppedLock)
        let value = stopped
        os_unfair_lock_unlock(&stoppedLock)
        return value
    }

    private func markStopped() -> Bool {
        os_unfair_lock_lock(&stoppedLock)
        let alreadyStopped = stopped
        stopped = true
        os_unfair_lock_unlock(&stoppedLock)
        return alreadyStopped
    }

    func microsecondsSinceAnchor(hostTime: UInt64) -> UInt64 {
        let signedDelta = Int64(bitPattern: hostTime) &- Int64(bitPattern: anchorHostTime)
        let clampedDelta: UInt64 = signedDelta < 0 ? 0 : UInt64(signedDelta)
        return clampedDelta &* timebaseNumer / timebaseDenom / 1000
    }

    func writeFrame(source: UInt8, timestampUs: UInt64, pcm: Data) {
        writeQueue.async { [weak self] in
            guard let self = self, !self.isStopped() else { return }
            var header = Data(capacity: 1 + 8 + 4)
            var src = source
            header.append(&src, count: 1)
            var ts = timestampUs.littleEndian
            withUnsafeBytes(of: &ts) { header.append($0.bindMemory(to: UInt8.self)) }
            var len = UInt32(pcm.count).littleEndian
            withUnsafeBytes(of: &len) { header.append($0.bindMemory(to: UInt8.self)) }
            do {
                try stdoutHandle.write(contentsOf: header)
                try stdoutHandle.write(contentsOf: pcm)
            } catch {
                self.shutdown(reason: "epipe")
            }
        }
    }

    func startSystemAudio() throws {
        let content = try awaitSync { completion in
            SCShareableContent.getExcludingDesktopWindows(
                false,
                onScreenWindowsOnly: false
            ) { content, error in
                completion(content, error)
            }
        }
        guard let display = content.displays.first else {
            failStartup("screen_recording_no_display")
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: writeQueue)
        let semaphore = DispatchSemaphore(value: 0)
        var startError: Error?
        stream.startCapture { error in
            startError = error
            semaphore.signal()
        }
        semaphore.wait()
        if let error = startError {
            throw error
        }
        self.stream = stream
    }

    func startMicrophone() throws -> String {
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)
        guard let tapFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        ) else {
            throw NSError(domain: "audio_capture", code: 1, userInfo: nil)
        }
        let converter = AVAudioConverter(from: inputFormat, to: tapFormat)
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, when in
            guard let self = self, let converter = converter else { return }
            let ratio = tapFormat.sampleRate / inputFormat.sampleRate
            let outputCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 1024)
            guard let outBuffer = AVAudioPCMBuffer(
                pcmFormat: tapFormat,
                frameCapacity: outputCapacity
            ) else { return }
            var error: NSError?
            var supplied = false
            converter.convert(to: outBuffer, error: &error) { _, status in
                if supplied {
                    status.pointee = .noDataNow
                    return nil
                }
                supplied = true
                status.pointee = .haveData
                return buffer
            }
            if let error = error {
                self.reportMicConverterError(error)
                return
            }
            let frameLength = Int(outBuffer.frameLength)
            guard frameLength > 0,
                  let channelData = outBuffer.floatChannelData?[0] else { return }
            let byteCount = frameLength * MemoryLayout<Float>.size
            let pcm = Data(bytes: channelData, count: byteCount)
            let timestampUs = self.microsecondsSinceAnchor(hostTime: when.hostTime)
            self.writeFrame(source: 0x02, timestampUs: timestampUs, pcm: pcm)
        }
        try audioEngine.start()
        let deviceName = inputNode.auAudioUnit.audioUnitName ?? "default"
        return deviceName
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, let dataBuffer = sampleBuffer.dataBuffer else { return }
        var totalLength: Int = 0
        var dataPointer: UnsafeMutablePointer<Int8>? = nil
        let status = CMBlockBufferGetDataPointer(
            dataBuffer,
            atOffset: 0,
            lengthAtOffsetOut: nil,
            totalLengthOut: &totalLength,
            dataPointerOut: &dataPointer
        )
        guard status == kCMBlockBufferNoErr, let pointer = dataPointer, totalLength > 0 else { return }
        let pcm = Data(bytes: pointer, count: totalLength)
        let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let hostTime = CMClockConvertHostTimeToSystemUnits(presentationTime)
        let timestampUs = microsecondsSinceAnchor(hostTime: hostTime)
        writeFrame(source: 0x01, timestampUs: timestampUs, pcm: pcm)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        shutdown(reason: "scstream_stopped")
    }

    func shutdown(reason: String) {
        if markStopped() { return }
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        if let stream = stream {
            stream.stopCapture { _ in }
        }
        flushStderr()
    }

    func reportMicConverterError(_ error: Error) {
        writeQueue.async { [weak self] in
            guard let self = self else { return }
            if self.micConverterErrorReported { return }
            self.micConverterErrorReported = true
            writeStderr("error: mic_converter_failed: \(error.localizedDescription)\n")
            flushStderr()
        }
    }
}

enum SyncError: Error {
    case missingContent
}

func awaitSync<T>(_ work: (@escaping (T?, Error?) -> Void) -> Void) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    var result: T?
    var resultError: Error?
    work { value, error in
        result = value
        resultError = error
        semaphore.signal()
    }
    semaphore.wait()
    if let error = resultError {
        throw error
    }
    guard let result = result else { throw SyncError.missingContent }
    return result
}

var captureSession: CaptureSession?

let signalQueue = DispatchQueue(label: "com.codeandsorts.chirp.capture-audio.signal")
var signalSources: [DispatchSourceSignal] = []

func installSignalHandlers() {
    for sig in [SIGTERM, SIGINT] {
        signal(sig, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: sig, queue: signalQueue)
        source.setEventHandler {
            captureSession?.shutdown(reason: "signal")
            flushStderr()
            exit(0)
        }
        source.resume()
        signalSources.append(source)
    }
}

func runMain() {
    signal(SIGPIPE, SIG_IGN)
    installSignalHandlers()

    let session = CaptureSession()
    captureSession = session

    let micSemaphore = DispatchSemaphore(value: 0)
    var micGranted = false
    writeStderr("capture: awaiting_permission\n")
    AVCaptureDevice.requestAccess(for: .audio) { granted in
        micGranted = granted
        micSemaphore.signal()
    }
    micSemaphore.wait()
    if !micGranted {
        failStartup("microphone_denied")
    }

    writeStderr("capture: awaiting_permission\n")
    do {
        try session.startSystemAudio()
    } catch {
        failStartup("screen_recording_denied")
    }

    let deviceName: String
    do {
        deviceName = try session.startMicrophone()
    } catch {
        failStartup("microphone_engine_failed")
    }

    writeStderr("capture: started\n")
    writeStderr("capture: sample_rate=16000 channels=1 format=float32\n")
    writeStderr("capture: system_audio=enabled\n")
    writeStderr("capture: microphone=enabled device=\"\(deviceName)\"\n")
    flushStderr()

    RunLoop.main.run()
}

runMain()
