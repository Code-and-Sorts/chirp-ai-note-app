import AVFoundation
import CoreGraphics
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
    private let sampleQueue = DispatchQueue(label: "com.codeandsorts.chirp.capture-audio.sample")
    private let anchorHostTime: UInt64
    private let timebaseNumer: UInt64
    private let timebaseDenom: UInt64
    private var stream: SCStream?
    private let audioEngine = AVAudioEngine()
    private var stoppedLock = os_unfair_lock()
    private var stopped = false
    private var micConverterErrorReported = false
    private var micEmittedSamples: UInt64 = 0
    private var micAnchorUs: UInt64 = 0
    private var micAnchored = false

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
        // Sync onto writeQueue so OS pipe back-pressure stalls the producer
        // (SCStream sampleQueue or AVAudioEngine tap thread) instead of
        // unbounded in-process buffering. Producers run on queues distinct
        // from writeQueue, so this never deadlocks.
        writeQueue.sync {
            if isStopped() { return }
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
                handleBrokenPipe()
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
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: sampleQueue)
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
            if !self.micAnchored {
                self.micAnchorUs = self.microsecondsSinceAnchor(hostTime: when.hostTime)
                self.micAnchored = true
            }
            let timestampUs = self.micAnchorUs &+ self.micEmittedSamples &* 1_000_000 / 16000
            self.micEmittedSamples &+= UInt64(frameLength)
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
        writeStderr("error: scstream_stopped: \(error.localizedDescription)\n")
        flushStderr()
        exit(1)
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
    flushStderr()
    // Write the sentinel before the screen-recording prompt fires — once macOS
    // shows it, future probes can read sentinel-present + preflight-false as
    // "denied" instead of "undetermined".
    try? FileManager.default.createDirectory(
        at: permissionSentinelURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    FileManager.default.createFile(atPath: permissionSentinelURL.path, contents: nil)
    // CGRequestScreenCaptureAccess is the documented way to surface the
    // Screen Recording TCC prompt and register this bundle with the
    // privacy database. Without it, SCShareableContent silently fails on
    // a fresh bucket and the user never sees the app appear in System
    // Settings → Privacy & Security → Screen Recording.
    if !CGRequestScreenCaptureAccess() {
        failStartup("screen_recording_denied")
    }
    do {
        try session.startSystemAudio()
    } catch {
        let nsError = error as NSError
        if nsError.code == -3801 {
            failStartup("screen_recording_denied")
        } else {
            failStartup("screen_recording_failed: \(nsError.localizedDescription)")
        }
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

// MARK: - Disclaim shim
//
// Without `responsibility_spawnattrs_setdisclaim`, macOS TCC inherits the
// "responsible process" from this binary's parent (the terminal that ran
// Python). Screen-recording / mic prompts then attribute to the terminal,
// not the bundled Chirp helper, and Chirp never appears in System Settings
// → Privacy & Security.
//
// To break the inheritance chain we re-spawn ourselves with the disclaim
// attribute set. The first invocation (no `CHIRP_CAPTURE_DISCLAIMED` env
// var) is just a launcher: it `posix_spawn`s a second copy of this same
// binary with the disclaim flag, forwards termination signals to the
// child, and propagates the child's exit status. The second invocation
// (env var set) skips this block and runs `runMain()`.
//
// Consolidating this into the Swift helper means the toolchain stays
// `swiftc`-only — no separate `clang` step in the Makefile — and the
// bundle ships one executable instead of two.

@_silgen_name("responsibility_spawnattrs_setdisclaim")
func responsibility_spawnattrs_setdisclaim(
    _ attrs: UnsafeMutablePointer<posix_spawnattr_t?>,
    _ disclaim: Int32
) -> Int32

private let disclaimSentinelEnv = "CHIRP_CAPTURE_DISCLAIMED"
private let childKillGraceSeconds: UInt32 = 2

private var disclaimedChildPid: pid_t = 0

// Sentinel marking that macOS has already shown the screen-recording permission
// dialog to this user. Lets --check-permissions distinguish "denied" from
// "never asked" — CGPreflightScreenCaptureAccess() returns Bool only.
let permissionSentinelURL: URL = FileManager.default
    .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    .appendingPathComponent("Chirp/.permission-prompted")

private func disclaimEscalateToSigkill(_ sig: Int32) {
    if disclaimedChildPid > 0 {
        kill(disclaimedChildPid, SIGKILL)
    }
}

private func disclaimForwardSignal(_ sig: Int32) {
    if disclaimedChildPid > 0 {
        kill(disclaimedChildPid, sig)
        // Arm an alarm so we SIGKILL the child if it ignores the
        // forwarded signal — otherwise Python's later SIGKILL would only
        // kill this launcher shim and orphan the child.
        signal(SIGALRM, disclaimEscalateToSigkill)
        alarm(childKillGraceSeconds)
    }
}

private func runDisclaimer() -> Int32 {
    var attrs: posix_spawnattr_t? = nil
    if posix_spawnattr_init(&attrs) != 0 {
        writeStderr("disclaim: posix_spawnattr_init failed\n")
        return 1
    }
    defer { posix_spawnattr_destroy(&attrs) }

    if responsibility_spawnattrs_setdisclaim(&attrs, 1) != 0 {
        // Non-fatal: continue without disclaim. TCC attribution will
        // fall back to the parent process; capture may still work if
        // the parent already holds the necessary grants.
        writeStderr("disclaim: setdisclaim failed; TCC may attribute to parent\n")
    }

    let argv = CommandLine.arguments.map { strdup($0) }
    var argvPointers: [UnsafeMutablePointer<CChar>?] = argv + [nil]

    let allowedEnvKeys: Set<String> = [
        "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "USER", "LOGNAME",
        disclaimSentinelEnv,
    ]
    let parentEnv = ProcessInfo.processInfo.environment
    var childEnv: [String: String] = [:]
    for key in allowedEnvKeys {
        if let value = parentEnv[key] {
            childEnv[key] = value
        }
    }
    childEnv[disclaimSentinelEnv] = "1"
    let envStrings = childEnv.map { strdup("\($0.key)=\($0.value)") }
    var envPointers: [UnsafeMutablePointer<CChar>?] = envStrings + [nil]

    defer {
        argv.forEach { free($0) }
        envStrings.forEach { free($0) }
    }

    var pid: pid_t = 0
    let executablePath = CommandLine.arguments[0]
    let spawnRc = argvPointers.withUnsafeMutableBufferPointer { argvBuf in
        envPointers.withUnsafeMutableBufferPointer { envBuf in
            posix_spawn(
                &pid, executablePath, nil, &attrs,
                argvBuf.baseAddress, envBuf.baseAddress
            )
        }
    }

    if spawnRc != 0 {
        let message = String(cString: strerror(spawnRc))
        writeStderr("disclaim: posix_spawn failed: \(message)\n")
        return 1
    }

    disclaimedChildPid = pid
    signal(SIGTERM, disclaimForwardSignal)
    signal(SIGINT, disclaimForwardSignal)
    signal(SIGHUP, disclaimForwardSignal)
    signal(SIGQUIT, disclaimForwardSignal)

    var status: Int32 = 0
    while waitpid(pid, &status, 0) < 0 {
        if errno != EINTR {
            return 1
        }
    }

    if (status & 0x7f) == 0 {
        return (status >> 8) & 0xff  // WIFEXITED → WEXITSTATUS
    }
    let termSig = status & 0x7f
    if termSig != 0 && termSig != 0x7f {
        // WIFSIGNALED — re-raise the same signal so the parent observes
        // the same exit cause.
        signal(termSig, SIG_DFL)
        kill(getpid(), termSig)
    }
    return 1
}

if CommandLine.arguments.contains("--check-permissions") {
    let screenRecordingState: String
    if CGPreflightScreenCaptureAccess() {
        screenRecordingState = "granted"
    } else if FileManager.default.fileExists(atPath: permissionSentinelURL.path) {
        screenRecordingState = "denied"
    } else {
        screenRecordingState = "undetermined"
    }

    let micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
    let microphoneState: String
    switch micStatus {
    case .authorized:
        microphoneState = "granted"
    case .notDetermined:
        microphoneState = "undetermined"
    default:
        microphoneState = "denied"
    }

    print("permission: screen_recording=\(screenRecordingState)")
    print("permission: microphone=\(microphoneState)")
    exit(0)
}

if ProcessInfo.processInfo.environment[disclaimSentinelEnv] == nil {
    exit(runDisclaimer())
}

runMain()
