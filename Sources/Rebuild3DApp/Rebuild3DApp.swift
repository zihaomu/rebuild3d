import SwiftUI
import Rebuild3DCore

@main
struct Rebuild3DApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var model = AppModel()

    var body: some Scene {
        Window("Rebuild3D", id: "main") {
            ContentView(model: model)
                .frame(minWidth: 960, minHeight: 620)
                .onAppear {
                    delegate.model = model
                    NSApp.setActivationPolicy(.regular)
                    NSApp.activate(ignoringOtherApps: true)
                    model.refreshDrafts()
                }
                .onOpenURL { model.openProject(at: $0) }
        }
        .defaultLaunchBehavior(.presented)
        .defaultSize(width: 1280, height: 800)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Project", action: model.newProject).keyboardShortcut("n")
                Button("Open Project…", action: model.openProject).keyboardShortcut("o")
                Divider()
                Button("Add Photos…", action: model.choosePhotos).keyboardShortcut("i")
                    .disabled(model.isBusy)
                Button("Discard Draft…", action: model.discardDraft).disabled(!model.isDraft || model.isBusy)
            }
            CommandGroup(replacing: .saveItem) {
                Button("Save Project", action: model.save).keyboardShortcut("s")
                    .disabled(model.project == nil || model.isBusy)
                Button("Export USDZ…", action: model.exportModel).keyboardShortcut("e", modifiers: [.command, .shift])
                    .disabled(model.project?.modelURL == nil || model.isBusy)
            }
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var model: AppModel?
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Activation must not depend on a view appearing during a background relaunch.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        model?.confirmLeavingProject() == false ? .terminateCancel : .terminateNow
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
}
