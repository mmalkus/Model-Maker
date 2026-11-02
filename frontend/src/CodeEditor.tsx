import { defaultKeymap } from '@codemirror/commands'
import { python } from '@codemirror/lang-python'
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { EditorState, Prec, type Extension } from '@codemirror/state'
import { EditorView, keymap } from '@codemirror/view'
import { basicSetup } from 'codemirror'
import { tags } from '@lezer/highlight'
import { useEffect, useRef } from 'react'
import { host as appHost } from './host'

const theme = EditorView.theme({
  '&': { fontSize: '11px', border: '1px solid var(--mm-border-strong)', borderRadius: '6px', backgroundColor: 'var(--mm-bg)' },
  '&.cm-focused': { outline: '2px solid var(--brand)', outlineOffset: '-1px' },
  '.cm-scroller': { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace', lineHeight: '1.5' },
  '.cm-gutters': { backgroundColor: 'var(--mm-bg-subtle)', border: 'none', borderRight: '1px solid var(--mm-bg-hover)' },
  '.cm-content': { padding: '6px 0' },
})

// Inside VS Code: the editor's own colours and token colours, so code reads
// on a dark or high-contrast theme too (see index.css for the rest).
const vscodeTheme: Extension = [
  EditorView.theme({
    '&': { color: 'var(--vscode-editor-foreground, var(--vscode-foreground))' },
    '.cm-scroller': { fontFamily: 'var(--vscode-editor-font-family)' },
    '.cm-content': { caretColor: 'var(--vscode-editorCursor-foreground)' },
    '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--vscode-editorCursor-foreground)' },
    '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': {
      backgroundColor: 'var(--vscode-editor-selectionBackground) !important',
    },
    '.cm-gutters': { color: 'var(--vscode-editorLineNumber-foreground)' },
    '.cm-activeLine': { backgroundColor: 'var(--vscode-editor-lineHighlightBackground, transparent)' },
    '.cm-activeLineGutter': {
      backgroundColor: 'var(--vscode-editor-lineHighlightBackground, transparent)',
      color: 'var(--vscode-editorLineNumber-activeForeground)',
    },
    '.cm-matchingBracket': { backgroundColor: 'var(--vscode-editorBracketMatch-background)' },
  }),
  Prec.high(
    syntaxHighlighting(
      HighlightStyle.define([
        { tag: [tags.keyword, tags.controlKeyword, tags.operatorKeyword, tags.definitionKeyword], color: 'var(--vscode-symbolIcon-keywordForeground)' },
        { tag: [tags.string, tags.special(tags.string)], color: 'var(--vscode-debugTokenExpression-string)' },
        { tag: [tags.number, tags.bool, tags.null], color: 'var(--vscode-debugTokenExpression-number)' },
        { tag: tags.comment, color: 'var(--vscode-descriptionForeground)', fontStyle: 'italic' },
        { tag: [tags.function(tags.variableName), tags.function(tags.definition(tags.variableName))], color: 'var(--vscode-symbolIcon-functionForeground)' },
        { tag: [tags.className, tags.definition(tags.className)], color: 'var(--vscode-symbolIcon-classForeground)' },
        { tag: [tags.propertyName, tags.attributeName], color: 'var(--vscode-symbolIcon-propertyForeground)' },
        { tag: tags.variableName, color: 'var(--vscode-foreground)' },
      ]),
    ),
  ),
]

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
  fill = false,
}: {
  value: string
  onChange?: (value: string) => void
  // Ctrl/Cmd-S inside the editor, so saving a block's code doesn't mean
  // reaching for the mouse mid-edit.
  onSave?: () => void
  readOnly?: boolean
  maxHeight?: number
  // Stretch to the parent's height and scroll inside the editor, instead of
  // growing up to maxHeight -- for a panel that already bounds the editor,
  // so there's one scrollbar rather than the panel's plus the editor's own.
  fill?: boolean
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
      appHost.kind === 'vscode' ? vscodeTheme : [],
      EditorView.lineWrapping,
      fill
        ? EditorView.theme({ '&': { height: '100%' } })
        : EditorView.theme({ '.cm-scroller': { maxHeight: `${maxHeight}px` } }),
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
  }, [readOnly, maxHeight, fill])

  useEffect(() => {
    const editor = view.current
    if (!editor || value === editor.state.doc.toString()) return
    editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: value } })
  }, [value])

  return <div ref={host} style={fill ? { height: '100%' } : undefined} />
}
