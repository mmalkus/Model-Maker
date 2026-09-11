import { defaultKeymap } from '@codemirror/commands'
import { python } from '@codemirror/lang-python'
import { EditorState, type Extension } from '@codemirror/state'
import { EditorView, keymap } from '@codemirror/view'
import { basicSetup } from 'codemirror'
import { useEffect, useRef } from 'react'

const theme = EditorView.theme({
  '&': { fontSize: '11px', border: '1px solid #d1d5db', borderRadius: '6px', backgroundColor: '#fff' },
  '&.cm-focused': { outline: '2px solid var(--brand)', outlineOffset: '-1px' },
  '.cm-scroller': { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace', lineHeight: '1.5' },
  '.cm-gutters': { backgroundColor: '#f9fafb', border: 'none', borderRight: '1px solid #f3f4f6' },
  '.cm-content': { padding: '6px 0' },
})

/** CodeMirror in a React wrapper. The editor owns its own document, so the
 * `value` prop is only pushed in when it differs from what's on screen --
 * which is what makes an AI draft (or switching to another block) land in
 * the editor without fighting the user's own typing for control of it. */
export function CodeEditor({
  value,
  onChange,
  onSave,
  readOnly = false,
  maxHeight = 320,
}: {
  value: string
  onChange?: (value: string) => void
  // Ctrl/Cmd-S inside the editor, so saving a block's code doesn't mean
  // reaching for the mouse mid-edit.
  onSave?: () => void
  readOnly?: boolean
  maxHeight?: number
}) {
  const host = useRef<HTMLDivElement | null>(null)
  const view = useRef<EditorView | null>(null)
  // Held in refs so changing a handler never forces the editor to be torn
  // down and rebuilt (which would drop cursor position and undo history).
  const onChangeRef = useRef(onChange)
  const onSaveRef = useRef(onSave)
  onChangeRef.current = onChange
  onSaveRef.current = onSave

  useEffect(() => {
    if (!host.current) return
    const extensions: Extension[] = [
      basicSetup,
      python(),
      theme,
      EditorView.lineWrapping,
      EditorView.theme({ '.cm-scroller': { maxHeight: `${maxHeight}px` } }),
      keymap.of([
        {
          key: 'Mod-s',
          preventDefault: true,
          run: () => {
            onSaveRef.current?.()
            return true
          },
        },
        ...defaultKeymap,
      ]),
      EditorView.updateListener.of((update) => {
        if (update.docChanged) onChangeRef.current?.(update.state.doc.toString())
      }),
    ]
    if (readOnly) extensions.push(EditorState.readOnly.of(true), EditorView.editable.of(false))

    const editor = new EditorView({ doc: value, extensions, parent: host.current })
    view.current = editor
    return () => {
      editor.destroy()
      view.current = null
    }
    // Rebuilt only when the editor's own configuration changes -- `value` is
    // synced by the effect below instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [readOnly, maxHeight])

  useEffect(() => {
    const editor = view.current
    if (!editor || value === editor.state.doc.toString()) return
    editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: value } })
  }, [value])

  return <div ref={host} />
}
