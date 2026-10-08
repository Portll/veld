// veld-enclave: a Touch ID factor for the veld keyring, held in the Secure Enclave.
//
// `enrol` creates a Secure Enclave P-256 key that only Touch ID (or, with
// `--policy presence`, the login password) can use, agrees a secret with it through
// an ephemeral key, proves the secret can be reproduced, and prints it once. veld
// wraps the keyring's master key under that secret. `unlock` reproduces the secret
// with the Secure Enclave key, which prompts for Touch ID. The private key never
// leaves the Secure Enclave; the record kept on disk is a handle only this Mac's
// Secure Enclave can use.
//
// Each command prints one JSON object on stdout. Failures print a message on stderr
// and exit 1.

import CryptoKit
import Foundation
import LocalAuthentication
import Security

struct Failure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct Record: Codable {
    let id: String
    let label: String?
    let createdAt: String
    let policy: String
    /// SecureEnclave.P256.KeyAgreement.PrivateKey.dataRepresentation, base64.
    let key: String
    /// The ephemeral public key the secret was agreed with, X9.63, base64.
    let ephemeral: String
}

let derivationInfo = Data("veld:secure-enclave:v1".utf8)

func directory() throws -> URL {
    if let dir = ProcessInfo.processInfo.environment["VELD_ENCLAVE_DIR"], !dir.isEmpty {
        return URL(fileURLWithPath: dir, isDirectory: true)
    }
    let support = try FileManager.default.url(
        for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    return support.appendingPathComponent("veld/enclave", isDirectory: true)
}

func recordURL(_ id: String) throws -> URL {
    guard id.range(of: "^se-[0-9a-f]{16}$", options: .regularExpression) != nil else {
        throw Failure("not a Secure Enclave factor id: \(id)")
    }
    return try directory().appendingPathComponent("\(id).json")
}

func save(_ record: Record) throws {
    let dir = try directory()
    try FileManager.default.createDirectory(
        at: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let url = try recordURL(record.id)
    guard FileManager.default.createFile(
        atPath: url.path, contents: try encoder.encode(record),
        attributes: [.posixPermissions: 0o600])
    else { throw Failure("could not write \(url.path)") }
}

func load(_ id: String) throws -> Record {
    let url = try recordURL(id)
    guard let data = FileManager.default.contents(atPath: url.path) else {
        throw Failure("no Secure Enclave factor \(id) on this Mac (\(url.path))")
    }
    return try JSONDecoder().decode(Record.self, from: data)
}

func derive(_ shared: SharedSecret, ephemeral: Data) -> Data {
    shared.hkdfDerivedSymmetricKey(
        using: SHA256.self, salt: ephemeral, sharedInfo: derivationInfo, outputByteCount: 32
    ).withUnsafeBytes { Data($0) }
}

func accessControl(_ policy: String) throws -> SecAccessControl {
    let flags: SecAccessControlCreateFlags
    switch policy {
    // biometryCurrentSet: adding or removing a fingerprint invalidates the key, so a
    // finger enrolled by someone who knows the login password cannot open it.
    case "biometry": flags = [.privateKeyUsage, .biometryCurrentSet]
    case "presence": flags = [.privateKeyUsage, .userPresence]
    default: throw Failure("unknown policy \(policy); use biometry or presence")
    }
    var error: Unmanaged<CFError>?
    guard let control = SecAccessControlCreateWithFlags(
        nil, kSecAttrAccessibleWhenUnlockedThisDeviceOnly, flags, &error)
    else { throw error!.takeRetainedValue() as Error }
    return control
}

func biometryReady() -> Bool {
    var error: NSError?
    return LAContext().canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error)
}

func reproduce(_ record: Record, reason: String) throws -> Data {
    guard let keyData = Data(base64Encoded: record.key),
        let ephemeral = Data(base64Encoded: record.ephemeral)
    else { throw Failure("the record for \(record.id) is corrupt") }
    let context = LAContext()
    context.localizedReason = reason
    let key = try SecureEnclave.P256.KeyAgreement.PrivateKey(
        dataRepresentation: keyData, authenticationContext: context)
    let peer = try P256.KeyAgreement.PublicKey(x963Representation: ephemeral)
    return derive(try key.sharedSecretFromKeyAgreement(with: peer), ephemeral: ephemeral)
}

func randomHex(_ count: Int) throws -> String {
    var bytes = [UInt8](repeating: 0, count: count)
    guard SecRandomCopyBytes(kSecRandomDefault, count, &bytes) == errSecSuccess else {
        throw Failure("no randomness")
    }
    return bytes.map { String(format: "%02x", $0) }.joined()
}

func enrol(label: String?, policy: String) throws -> [String: Any] {
    guard SecureEnclave.isAvailable else { throw Failure("this Mac has no Secure Enclave") }
    if policy == "biometry" && !biometryReady() {
        throw Failure("Touch ID is not set up; enrol a fingerprint or use --policy presence")
    }
    let key = try SecureEnclave.P256.KeyAgreement.PrivateKey(accessControl: try accessControl(policy))
    let ephemeral = P256.KeyAgreement.PrivateKey()
    let ephemeralPublic = ephemeral.publicKey.x963Representation
    let secret = derive(
        try ephemeral.sharedSecretFromKeyAgreement(with: key.publicKey), ephemeral: ephemeralPublic)
    let record = Record(
        id: "se-\(try randomHex(8))",
        label: label,
        createdAt: ISO8601DateFormatter().string(from: Date()),
        policy: policy,
        key: key.dataRepresentation.base64EncodedString(),
        ephemeral: ephemeralPublic.base64EncodedString())
    guard try reproduce(record, reason: "confirm Touch ID for the veld keyring") == secret else {
        throw Failure("the Secure Enclave did not reproduce the secret; nothing was enrolled")
    }
    try save(record)
    return [
        "id": record.id, "label": label ?? NSNull(), "policy": policy,
        "secret": secret.base64EncodedString(),
    ]
}

func unlock(id: String, reason: String) throws -> [String: Any] {
    let record = try load(id)
    return [
        "id": record.id, "label": record.label ?? NSNull(),
        "secret": try reproduce(record, reason: reason).base64EncodedString(),
    ]
}

func list() throws -> [String: Any] {
    let dir = try directory()
    let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
    let records = try names.filter { $0.hasSuffix(".json") }.sorted().map { name -> [String: Any] in
        let record = try load(String(name.dropLast(5)))
        return [
            "id": record.id, "label": record.label ?? NSNull(), "policy": record.policy,
            "created_at": record.createdAt,
        ]
    }
    return ["factors": records]
}

func remove(id: String) throws -> [String: Any] {
    try FileManager.default.removeItem(at: try recordURL(id))
    return ["removed": id]
}

func option(_ name: String, in args: [String]) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

let usage = """
    usage: veld-enclave probe
           veld-enclave enrol [--label TEXT] [--policy biometry|presence]
           veld-enclave unlock --id ID [--reason TEXT]
           veld-enclave list
           veld-enclave delete --id ID
    """

let args = Array(CommandLine.arguments.dropFirst())
do {
    let result: [String: Any]
    switch args.first {
    case "probe":
        result = ["secure_enclave": SecureEnclave.isAvailable, "touch_id": biometryReady()]
    case "enrol":
        result = try enrol(
            label: option("--label", in: args), policy: option("--policy", in: args) ?? "biometry")
    case "unlock":
        guard let id = option("--id", in: args) else { throw Failure(usage) }
        result = try unlock(id: id, reason: option("--reason", in: args) ?? "unlock the veld keyring")
    case "list":
        result = try list()
    case "delete":
        guard let id = option("--id", in: args) else { throw Failure(usage) }
        result = try remove(id: id)
    default:
        throw Failure(usage)
    }
    let json = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    FileHandle.standardOutput.write(json)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    FileHandle.standardError.write(Data("veld-enclave: \(error)\n".utf8))
    exit(1)
}
