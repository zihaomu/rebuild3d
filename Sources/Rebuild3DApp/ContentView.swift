import ImageIO
import Rebuild3DCore
import SwiftUI

struct ContentView: View {
    @Bindable var model: AppModel
    @State private var showInspector = true
    @State private var showDiagnostics = false

    var body: some View {
        VStack(spacing: 0) {
            HSplitView {
                photoSidebar.frame(minWidth: 200, idealWidth: 240, maxWidth: 340)
                viewport.frame(minWidth: 380, maxWidth: .infinity, maxHeight: .infinity)
                if showInspector { inspector.frame(minWidth: 250, idealWidth: 290, maxWidth: 380) }
            }
            Divider()
            HStack {
                Text(model.status).lineLimit(2)
                Spacer()
                Text("\(model.project?.manifest.photos.count ?? 0) 张照片")
                if model.isDraft { Text("可恢复草稿").foregroundStyle(.secondary) }
                if model.isDirty { Text("Unsaved changes").foregroundStyle(.orange) }
                if !model.diagnostics.isEmpty {
                    Button("Diagnostics") { showInspector = true; showDiagnostics = true }
                }
            }
            .font(.caption).padding(10)
        }
        .navigationTitle(model.project?.manifest.name ?? "Rebuild3D")
        .windowDismissBehavior(model.isBusy ? .disabled : .enabled)
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                Button(action: model.newProject) { Label("New", systemImage: "doc.badge.plus") }.disabled(model.isBusy)
                Button(action: model.openProject) { Label("Open", systemImage: "folder") }.disabled(model.isBusy)
                Button(action: model.choosePhotos) { Label("添加照片", systemImage: "photo.badge.plus") }
                    .disabled(model.isBusy)
                if !model.recoverableDrafts.isEmpty {
                    Menu {
                        ForEach(model.recoverableDrafts) { draft in
                            Button("\(draft.name) · \(draft.photoCount) photos · \(draft.modifiedAt.formatted(date: .abbreviated, time: .shortened))") {
                                model.openProject(at: draft.directory)
                            }
                        }
                    } label: { Label("恢复草稿", systemImage: "clock.arrow.circlepath") }
                    .disabled(model.isBusy)
                }
            }
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    if model.isReconstructing { model.cancel() } else { model.reconstruct() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: model.isReconstructing ? "stop.circle" : "cube.transparent")
                        Text(model.isReconstructing ? "取消" : "生成模型")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isReconstructing ? model.isCancelling : !model.canReconstruct)
                .help(model.isReconstructing ? "取消并保留有效进度。" : model.inputCheck.blockingReason ?? "自动准备组件、生成主体、铺设照片纹理并保存。")
                Button(action: model.save) { Label("Save", systemImage: "square.and.arrow.down") }
                    .disabled(model.project == nil || model.isBusy)
                Button(action: model.exportModel) { Label("Export USDZ", systemImage: "square.and.arrow.up") }
                    .disabled(model.project?.modelURL == nil || model.isBusy)
                Button { showInspector.toggle() } label: { Label("Inspector", systemImage: "sidebar.right") }
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard !model.isBusy else { return false }
            model.importPhotos(urls)
            return true
        }
        .alert("Unable to Complete Operation", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK") { model.errorMessage = nil }
        } message: { Text(model.errorMessage ?? "") }
    }

    private var photoSidebar: some View {
        let inputCheck = model.inputCheck
        return VStack(alignment: .leading, spacing: 0) {
            Text("PHOTOS").font(.caption.bold()).foregroundStyle(.secondary).padding()
            List(selection: $model.selectedPhotoID) {
                ForEach(model.project?.manifest.photos ?? []) { photo in
                    HStack(spacing: 10) {
                        if let project = model.project {
                            PhotoPreview(url: project.directory.appendingPathComponent(photo.thumbnailPath), maxPixelSize: 160)
                                .frame(width: 54, height: 54).clipped()
                        }
                        VStack(alignment: .leading) {
                            Text(photo.originalName).lineLimit(1)
                            Text("\(photo.pixelWidth) × \(photo.pixelHeight)").font(.caption).foregroundStyle(.secondary)
                            if let issue = inputCheck.photoIssues[photo.id] {
                                Label(issue, systemImage: "exclamationmark.triangle")
                                    .font(.caption).foregroundStyle(.orange)
                            }
                        }
                    }.tag(photo.id)
                }
                let problems = model.importIssues.filter { $0.kind == .unreadable || $0.kind == .unsupported }
                if !problems.isEmpty {
                    Section("Needs Attention") {
                        ForEach(problems) { issue in
                            VStack(alignment: .leading, spacing: 4) {
                                Label(issue.filename, systemImage: "exclamationmark.triangle")
                                Text(issue.reason).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            Button("Remove Selected Photo", systemImage: "minus.circle", action: model.removeSelectedPhoto)
                .disabled(model.selectedPhotoID == nil || model.isBusy).padding()
        }
    }

    private var viewport: some View {
        ZStack {
            if let url = model.displayedModelURL {
                ModelViewport(url: url, resetID: model.resetViewID) { model.errorMessage = $0 }
                VStack {
                    HStack {
                        Text("Free View").font(.caption).padding(8).background(.regularMaterial, in: Capsule())
                        if model.project?.manifest.model?.approximation != nil {
                            Text("Approximate").font(.caption.bold()).foregroundStyle(.orange)
                                .padding(8).background(.regularMaterial, in: Capsule())
                            Picker("查看来源", selection: $model.appearance) {
                                Text(ModelAppearance.photos.rawValue).tag(ModelAppearance.photos)
                                Text(ModelAppearance.geometry.rawValue).tag(ModelAppearance.geometry)
                                if model.project?.textureSourcesModelURL != nil {
                                    Text(ModelAppearance.texture.rawValue).tag(ModelAppearance.texture)
                                }
                            }.pickerStyle(.menu).fixedSize().padding(8)
                                .background(.regularMaterial, in: Capsule())
                        }
                        Spacer()
                        Button("Reset View", systemImage: "arrow.counterclockwise") { model.resetViewID = UUID() }
                            .buttonStyle(.bordered).background(.regularMaterial, in: Capsule())
                    }
                    if model.project?.modelIsOutdated == true {
                        Label("上次结果，当前照片尚未生成", systemImage: "clock.arrow.circlepath")
                            .font(.callout.bold()).foregroundStyle(.orange)
                            .padding(10).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                    }
                    Spacer()
                    if model.project?.manifest.model?.approximation != nil {
                        Text(model.sourceDescription).multilineTextAlignment(.center)
                            .font(.caption).padding(8).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                    }
                    Text("Drag to orbit · Scroll to zoom · Shift-drag to pan")
                        .font(.caption).padding(8).background(.regularMaterial, in: Capsule())
                }.padding()
            } else if !model.isBusy {
                ContentUnavailableView {
                    Label("从照片生成三维模型", systemImage: "cube.transparent")
                } description: {
                    Text((model.project?.manifest.photos.isEmpty ?? true)
                         ? "添加同一物体不同角度的照片，再点击生成模型。"
                         : model.inputCheck.blockingReason ?? "已准备好。生成会自动处理主体和贴图，可能需要较长时间。")
                } actions: {
                    if model.project?.manifest.photos.isEmpty ?? true {
                        Button("添加照片…", action: model.choosePhotos).buttonStyle(.borderedProminent).disabled(model.isBusy)
                    } else {
                        Button("生成模型", action: model.reconstruct).buttonStyle(.borderedProminent).disabled(!model.canReconstruct)
                        if model.inputCheck.blockingReason != nil {
                            Button("添加照片…", action: model.choosePhotos).disabled(model.isBusy)
                        }
                    }
                }
            }
            if model.isBusy {
                VStack(spacing: 12) {
                    if model.isReconstructing, let progress = model.progress {
                        ProgressView(value: progress).frame(width: 260)
                        Text(progress, format: .percent.precision(.fractionLength(0)))
                    } else { ProgressView() }
                    Text(model.status).font(.callout).multilineTextAlignment(.center)
                    if model.isReconstructing, let started = model.generationStartedAt {
                        TimelineView(.periodic(from: started, by: 1)) { context in
                            let seconds = max(0, Int(context.date.timeIntervalSince(started)))
                            Text("已用时 \(seconds / 60) 分 \(seconds % 60) 秒")
                                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        }
                        Text("可以取消，稍后从已完成阶段继续。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .padding(24).frame(maxWidth: 360).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
            }
        }
    }

    private var inspector: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                DisclosureGroup("高级选项") {
                    Picker("Quality", selection: Binding(get: { model.project?.manifest.settings.quality ?? .reduced }, set: { model.setQuality($0) })) {
                        ForEach(ReconstructionQuality.allCases, id: \.self) { quality in
                            Text(quality == .reduced ? "Recommended" : "More Detail").tag(quality)
                        }
                    }.disabled(model.project == nil || model.isBusy)
                    Text("少量照片使用自动设置；质量选项适用于常规重建。")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("Isolate object from background", isOn: Binding(get: { model.project?.manifest.settings.objectMasking ?? true }, set: { model.setMasking($0) }))
                        .disabled(model.project == nil || model.isBusy)
                    Text("More detail may use more memory and take longer.").font(.caption).foregroundStyle(.secondary)
                    Button("导入已有研究结果…", action: model.loadApproximateResult)
                        .disabled(model.isBusy || (model.project?.manifest.photos.isEmpty ?? true))
                    Button("查看本地组件许可…", action: model.showComponentNotices)
                }
                DisclosureGroup("Photo Tips") {
                    Text("Use a stationary, textured object in even light. Keep the same lens and zoom. Move around the object at different heights, keeping most of each photo in common with its neighbors. Add the HEIC or JPEG files directly.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let reason = model.inputCheck.blockingReason,
                   !(model.project?.manifest.photos.isEmpty ?? true) || !ReconstructionEngine.isSupported {
                    Label(reason, systemImage: "exclamationmark.triangle")
                        .font(.callout).foregroundStyle(.orange)
                }
                if model.project?.modelIsOutdated == true {
                    Label("上次结果，当前照片尚未生成。点击生成模型以更新。", systemImage: "info.circle")
                        .font(.caption).foregroundStyle(.orange)
                }
                Divider()
                if model.appearance == .texture, !model.sourceLegend.isEmpty {
                    Text("颜色来源").font(.headline)
                    ForEach(model.sourceLegend) { item in
                        HStack {
                            Circle().fill(Color(red: item.rgb[0] / 255, green: item.rgb[1] / 255, blue: item.rgb[2] / 255))
                                .frame(width: 12, height: 12)
                            Text(item.code == 0 ? "外观填充（推测）" : item.meaning).font(.caption)
                        }
                    }
                }
                if let photo = model.selectedPhoto, let project = model.project {
                    Text("Original Photo").font(.headline)
                    PhotoPreview(url: project.directory.appendingPathComponent(photo.imagePath), maxPixelSize: 1200)
                        .frame(maxWidth: .infinity).frame(height: 240)
                    Text(photo.originalName).font(.caption).textSelection(.enabled)
                }
                if !model.importIssues.isEmpty {
                    DisclosureGroup("Import Details (\(model.importIssues.count))") {
                        ForEach(model.importIssues) { issue in
                            VStack(alignment: .leading, spacing: 4) {
                                Text("\(issue.filename) · \(issue.kind.rawValue)").font(.caption.bold())
                                Text(issue.reason).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                if !model.diagnostics.isEmpty {
                    DisclosureGroup("Diagnostics", isExpanded: $showDiagnostics) {
                        Text(model.diagnostics.joined(separator: "\n")).font(.caption.monospaced()).textSelection(.enabled)
                    }
                }
            }.padding()
        }
    }
}

private struct PhotoPreview: View {
    let url: URL
    let maxPixelSize: Int
    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image { Image(nsImage: image).resizable().scaledToFit() }
            else { Image(systemName: "photo").foregroundStyle(.secondary) }
        }
        .task(id: url) {
            image = nil
            let loaded = await Task.detached(priority: .utility) {
                try? PhotoPreviewDecoder.image(at: url, maxPixelSize: maxPixelSize)
            }.value
            guard !Task.isCancelled else { return }
            image = loaded.map { NSImage(cgImage: $0, size: .zero) }
        }
    }
}
