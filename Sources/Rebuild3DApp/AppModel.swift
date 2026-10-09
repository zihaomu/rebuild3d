import AppKit
import Observation
import Rebuild3DCore
import UniformTypeIdentifiers

enum ModelAppearance: String, CaseIterable {
    case photos = "照片颜色", geometry = "推测几何", texture = "颜色来源"
}

struct TextureSourceLegend: Decodable, Identifiable {
    let code: Int
    let rgb: [Double]
    let photoID: UUID?
    let meaning: String
    var id: Int { code }
}

@MainActor @Observable
final class AppModel {
    var project: Project?
    var selectedPhotoID: UUID?
    var isBusy = false
    var isReconstructing = false
    var isCancelling = false
    var isDirty = false
    var progress: Double?
    var status = "添加同一物体不同角度的照片，然后点击生成模型。"
    var errorMessage: String?
    var importIssues: [ImportIssue] = []
    var resetViewID = UUID()
    var appearance = ModelAppearance.photos
    var generationStartedAt: Date?
    var sourceLegend: [TextureSourceLegend] = []
    var diagnostics: [String] = []
    var recoverableDrafts: [DraftSummary] = []
    let drafts = DraftProjectStore()
    let engine = GenerationCoordinator()

    var selectedPhoto: PhotoRecord? { project?.manifest.photos.first { $0.id == selectedPhotoID } }
    var inputCheck: ReconstructionInputCheck { GenerationCoordinator.checkInput(project?.manifest.photos ?? []) }
    var canReconstruct: Bool { !isBusy && inputCheck.blockingReason == nil }
    var isDraft: Bool { project.map { drafts.contains($0.directory) } ?? false }
    var displayedModelURL: URL? {
        switch appearance {
        case .photos: project?.modelURL
        case .geometry: project?.provenanceModelURL ?? project?.modelURL
        case .texture: project?.textureSourcesModelURL ?? project?.modelURL
        }
    }
    var sourceDescription: String {
        switch appearance {
        case .photos: "形状为推测，颜色来自照片；不可见区域采用保守填充。比例不代表真实尺寸。"
        case .geometry: "橙色：深度推测 · 紫色：轮廓补全。所有几何均为推测，并非实测。"
        case .texture: "彩色区域来自不同原照片；灰色为外观填充。照片颜色不代表几何已获实测验证。"
        }
    }

    func refreshDrafts() {
        let store = drafts
        Task {
            do {
                let listed = try await Task.detached { try store.list() }.value
                recoverableDrafts = listed.filter { $0.directory != project?.directory }
            } catch { errorMessage = "Drafts could not be listed: \(error.localizedDescription)" }
        }
    }

    func newProject() {
        guard !isBusy, confirmLeavingProject() else { return }
        project = nil
        appearance = .photos
        sourceLegend = []
        selectedPhotoID = nil
        importIssues = []
        diagnostics = []
        isDirty = false
        progress = nil
        status = "添加同一物体不同角度的照片，然后点击生成模型。"
        refreshDrafts()
    }

    func openProject() {
        guard !isBusy, confirmLeavingProject() else { return }
        let panel = NSOpenPanel()
        panel.title = "Open Rebuild3D Project"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.treatsFilePackagesAsDirectories = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        openProject(at: url, confirmed: true)
    }

    func openProject(at url: URL, confirmed: Bool = false) {
        guard !isBusy, confirmed || confirmLeavingProject() else { return }
        perform {
            let opened = try await Task.detached {
                let opened = try ProjectStore.open(url)
                for photo in opened.manifest.photos {
                    let thumbnail = try ProjectStore.resolve(photo.thumbnailPath, in: opened.directory)
                    if !FileManager.default.fileExists(atPath: thumbnail.path) {
                        // Thumbnails are disposable; failure to regenerate one must not hide a saved model.
                        try? PhotoImporter.regenerateThumbnail(for: photo, in: opened.directory)
                    }
                }
                return opened
            }.value
            self.install(opened)
        }
    }

    private func install(_ project: Project) {
        self.project = project
        appearance = .photos
        refreshSourceLegend()
        selectedPhotoID = project.manifest.photos.first?.id
        importIssues = []
        isDirty = project.needsMigrationSave
        progress = nil
        diagnostics = []
        status = project.modelURL == nil ? "点击生成模型。中断过的任务会从有效阶段继续。" : "已恢复保存的模型。"
        if project.manifest.model?.approximation != nil { status = "已恢复近似模型，可查看推测区域和颜色来源。" }
        if project.modelIsOutdated { status = "上次结果，当前照片尚未生成。" }
        if isDraft { status += " 草稿已恢复。" }
        refreshDrafts()
    }

    private func refreshSourceLegend() {
        struct Sources: Decodable { let sourceDisplayColors: [TextureSourceLegend] }
        sourceLegend = []
        guard let project, let reference = project.manifest.model?.approximation,
              let bundle = try? ProjectStore.resolve(reference.bundlePath, in: project.directory),
              let url = try? ProjectStore.resolve("texture-provenance.json", in: bundle.deletingLastPathComponent()),
              let data = try? Data(contentsOf: url), let sources = try? JSONDecoder().decode(Sources.self, from: data) else { return }
        sourceLegend = sources.sourceDisplayColors.filter { $0.rgb.count == 3 }
    }

    func choosePhotos() {
        guard !isBusy else { return }
        let panel = NSOpenPanel()
        panel.title = "Import Photos or a Photo Folder"
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        guard panel.runModal() == .OK else { return }
        importPhotos(panel.urls)
    }

    func showComponentNotices() {
        guard let url = Bundle.main.resourceURL?.appendingPathComponent("GenerationRuntime/NOTICE.md"),
              FileManager.default.fileExists(atPath: url.path) else {
            errorMessage = "此应用包未包含生成组件的许可文件。"; return
        }
        NSWorkspace.shared.open(url)
    }

    func loadApproximateResult() {
        guard let project, !isBusy else { return }
        let panel = NSOpenPanel()
        panel.title = "Load Approximate Reconstruction Result"
        panel.message = "Choose the research result folder. Original-photo identities and inferred-region records will be checked."
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let directory = panel.url else { return }
        status = "Checking and saving approximate reconstruction…"
        perform {
            let updated = try await Task.detached { try ApproximateResultStore.importResult(from: directory, into: project) }.value
            self.install(updated)
            self.status = "Approximate reconstruction saved. Use Sources to inspect inferred and completed regions."
            self.resetViewID = UUID()
        }
    }

    func importPhotos(_ urls: [URL]) {
        guard !isBusy, !urls.isEmpty else { return }
        status = "Importing photos…"
        perform {
            if self.project == nil {
                let store = self.drafts
                self.project = try await Task.detached { try store.create() }.value
            }
            guard let project = self.project else { return }
            let result = try await Task.detached {
                try PhotoImporter.importPhotos(from: urls, into: project.directory, existingPhotos: project.manifest.photos)
            }.value
            self.project?.manifest.photos.append(contentsOf: result.photos)
            self.importIssues = result.issues
            if !result.photos.isEmpty {
                self.markDirty()
                self.selectedPhotoID = result.photos.first?.id
                if self.isDraft, let updated = self.project {
                    try await Task.detached { try ProjectStore.save(updated) }.value
                    self.isDirty = false
                }
            }
            self.status = result.summary
        }
    }

    func removeSelectedPhoto() {
        guard !isBusy, let selectedPhotoID else { return }
        project?.manifest.photos.removeAll { $0.id == selectedPhotoID }
        self.selectedPhotoID = project?.manifest.photos.first?.id
        markDirty()
        status = "Photo removed from the input list. Save to keep this change."
        // Retain source bytes until an explicit future cleanup so failed saves never lose originals.
        saveDraftChanges()
    }

    func setQuality(_ quality: ReconstructionQuality) {
        guard !isBusy else { return }
        project?.manifest.settings.quality = quality
        markDirty()
        saveDraftChanges()
    }

    func setMasking(_ enabled: Bool) {
        guard !isBusy else { return }
        project?.manifest.settings.objectMasking = enabled
        markDirty()
        saveDraftChanges()
    }

    private func markDirty() {
        project?.manifest.modifiedAt = Date()
        isDirty = true
    }

    private func saveDraftChanges() {
        guard isDraft, let project, !isBusy else { return }
        perform {
            try await Task.detached { try ProjectStore.save(project) }.value
            self.isDirty = false
            self.status = "Draft saved for recovery."
        }
    }

    func save() {
        guard let project, !isBusy else { return }
        if isDraft {
            let panel = NSSavePanel()
            panel.title = "Save Rebuild3D Project"
            panel.nameFieldStringValue = "\(project.manifest.name).rebuild3d"
            panel.allowedContentTypes = [UTType(exportedAs: "org.rebuild3d.project", conformingTo: .package)]
            panel.canCreateDirectories = true
            guard panel.runModal() == .OK, let destination = panel.url else { return }
            let store = drafts
            perform {
                let saved = try await Task.detached { try store.saveAs(project, to: destination) }.value
                self.install(saved)
                do {
                    try await Task.detached { try store.discard(project.directory) }.value
                    self.status = "Project saved."
                } catch {
                    self.status = "Project saved. Its recovery copy could not be removed."
                    self.diagnostics.append(error.localizedDescription)
                }
                self.refreshDrafts()
            }
            return
        }
        perform {
            try await Task.detached { try ProjectStore.save(project) }.value
            self.project?.loadedFormatVersion = ProjectManifest.currentVersion
            self.isDirty = false
            self.status = "Project saved."
        }
    }

    func discardDraft() {
        guard isDraft, let project, !isBusy else { return }
        let alert = NSAlert()
        alert.messageText = "Discard this draft?"
        alert.informativeText = "The draft's copied photos and reconstructed results will be deleted. Source photos outside the draft are not affected."
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: "Discard Draft")
        guard alert.runModal() == .alertSecondButtonReturn else { return }
        let store = drafts
        perform {
            try await Task.detached { try store.discard(project.directory) }.value
            self.project = nil
            self.selectedPhotoID = nil
            self.importIssues = []
            self.isDirty = false
            self.status = "Draft discarded. Add photos to begin again."
            self.refreshDrafts()
        }
    }

    func reconstruct() {
        guard let project, canReconstruct else { return }
        isReconstructing = true
        isCancelling = false
        progress = nil
        diagnostics = []
        generationStartedAt = Date()
        status = "正在准备生成模型…"
        perform {
            defer { self.isReconstructing = false; self.isCancelling = false }
            try await Task.detached { try ProjectStore.save(project) }.value
            self.isDirty = false
            guard !self.isCancelling else { throw CancellationError() }
            let updated = try await self.engine.run(project: project) { event in
                await self.receive(event)
            }
            self.project = updated
            self.appearance = .photos
            self.refreshSourceLegend()
            self.progress = 1
            self.status = "本次模型已生成并保存，可查看或导出。"
            self.resetViewID = UUID()
        }
    }

    private func receive(_ event: ReconstructionEvent) {
        switch event {
        case .stage(let stage):
            progress = nil
            if !isCancelling { status = stage.message }
        case .progress(let value): progress = value
        case .message(let message): if !isCancelling { status = message }
        case .diagnostic(let detail):
            if diagnostics.last != detail { diagnostics.append(detail) }
        }
    }

    func cancel() {
        guard isReconstructing, !isCancelling else { return }
        isCancelling = true
        status = "正在停止生成并保存有效进度…"
        Task { await engine.cancel() }
    }

    func exportModel() {
        guard let project, project.modelURL != nil, !isBusy else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(project.manifest.name).usdz"
        panel.allowedContentTypes = [.usdz]
        if project.modelIsOutdated { panel.message = "导出上次已保存的模型。当前照片尚未生成，导出不会包含本次输入修改。" }
        if project.manifest.model?.approximation != nil {
            panel.message = (project.modelIsOutdated ? "导出上次结果，当前照片尚未生成。\n" : "") + "同时导出 USDZ 和来源附件目录，保留照片身份、推测区域和颜色来源。"
        }
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        perform {
            try await Task.detached { try ProjectStore.exportModel(project, to: destination) }.value
            self.status = project.manifest.model?.approximation == nil
                ? "USDZ exported with its embedded materials and textures."
                : "Approximate USDZ exported with its companion source-region and provenance folder."
        }
    }

    /// Used by both document replacement and application termination.
    func confirmLeavingProject() -> Bool {
        if isBusy {
            let alert = NSAlert()
            alert.messageText = "An operation is still running"
            alert.informativeText = "Wait for it to finish, or cancel reconstruction before closing the project."
            alert.runModal()
            return false
        }
        guard isDirty, let project else { return true }
        if isDraft {
            do { try ProjectStore.save(project); isDirty = false; return true }
            catch { errorMessage = "The draft could not be saved for recovery: \(error.localizedDescription)"; return false }
        }
        let alert = NSAlert()
        alert.messageText = "Save changes to \(project.manifest.name)?"
        alert.informativeText = "Unsaved input and settings changes will be lost."
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")
        alert.addButton(withTitle: "Discard Changes")
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            // The manifest is small; a synchronous write keeps termination transactional.
            do { try ProjectStore.save(project); isDirty = false; return true }
            catch { errorMessage = error.localizedDescription; return false }
        case .alertThirdButtonReturn: return true
        default: return false
        }
    }

    private func perform(_ action: @escaping @MainActor () async throws -> Void) {
        isBusy = true
        Task {
            defer { isBusy = false }
            do { try await action() }
            catch is CancellationError { status = "已取消生成，照片和上次结果已保留。再次点击生成可从有效阶段继续。" }
            catch {
                errorMessage = error.localizedDescription
                status = "本次操作未完成，照片和已保存结果已保留。"
            }
        }
    }
}
