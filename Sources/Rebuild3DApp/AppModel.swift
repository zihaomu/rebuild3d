import AppKit
import Observation
import Rebuild3DCore
import UniformTypeIdentifiers

@MainActor @Observable
final class AppModel {
    var project: Project?
    var selectedPhotoID: UUID?
    var isBusy = false
    var isReconstructing = false
    var isCancelling = false
    var isDirty = false
    var progress: Double?
    var status = "Add photos of the same object from different angles."
    var errorMessage: String?
    var importIssues: [ImportIssue] = []
    var resetViewID = UUID()
    var showProvenance = false
    var diagnostics: [String] = []
    var recoverableDrafts: [DraftSummary] = []
    let drafts = DraftProjectStore()
    let engine = ReconstructionEngine()

    var selectedPhoto: PhotoRecord? { project?.manifest.photos.first { $0.id == selectedPhotoID } }
    var inputCheck: ReconstructionInputCheck { ReconstructionEngine.checkInput(project?.manifest.photos ?? []) }
    var canReconstruct: Bool { !isBusy && inputCheck.blockingReason == nil }
    var isDraft: Bool { project.map { drafts.contains($0.directory) } ?? false }
    var displayedModelURL: URL? { showProvenance ? project?.provenanceModelURL ?? project?.modelURL : project?.modelURL }

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
        showProvenance = false
        selectedPhotoID = nil
        importIssues = []
        diagnostics = []
        isDirty = false
        progress = nil
        status = "Add photos of the same object from different angles."
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
        showProvenance = false
        selectedPhotoID = project.manifest.photos.first?.id
        importIssues = []
        isDirty = project.needsMigrationSave
        progress = nil
        diagnostics = []
        status = project.modelURL == nil ? "Add photos, then start reconstruction with the recommended settings." : "Saved model restored."
        if project.manifest.model?.approximation != nil { status = "Approximate model restored with its inferred-region records." }
        if isDraft { status += " Draft recovered." }
        refreshDrafts()
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
        status = "Preparing reconstruction…"
        perform {
            defer { self.isReconstructing = false; self.isCancelling = false }
            try await Task.detached { try ProjectStore.save(project) }.value
            self.isDirty = false
            guard !self.isCancelling else { throw CancellationError() }
            let updated = try await self.engine.run(project: project) { event in
                await self.receive(event)
            }
            self.project = updated
            self.showProvenance = false
            self.progress = 1
            self.status = "Textured model saved. Inspect the result and export USDZ."
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
        status = "Cancelling reconstruction…"
        Task { await engine.cancel() }
    }

    func exportModel() {
        guard let project, project.modelURL != nil, !isBusy else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "\(project.manifest.name).usdz"
        panel.allowedContentTypes = [.usdz]
        if project.manifest.model?.approximation != nil {
            panel.message = "Exports the USDZ and a companion .rebuild3d-result folder containing source regions, original-photo identities and research records."
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
            catch is CancellationError { status = "Reconstruction cancelled. Any previously saved model is preserved." }
            catch {
                errorMessage = error.localizedDescription
                status = "Operation failed. Review the error and retry."
            }
        }
    }
}
