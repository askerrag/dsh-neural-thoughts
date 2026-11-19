window.__ModuleLoader__.load({
  id: '@local/neural-thoughts',
  factory(require) {
    'use strict'

    const React = require('react')
    const h = React.createElement
    const {
      useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore,
    } = React

    const NS = 'neuralThoughts'
    const MAX_NODES = 320
    const CLIP_TEXT = 16000
    const TAU = Math.PI * 2
    const IS_MAC = (() => {
      try {
        const platform = String(navigator.platform || '')
        const agent = String(navigator.userAgent || '')
        return /mac/i.test(platform) || /Macintosh|Mac OS X/i.test(agent)
      } catch { return false }
    })()

    /* ------------------------------------------------------------------ *
     *  Sectors of the artificial brain
     * ------------------------------------------------------------------ */

    const SECTORS = {
      prefrontal: { color: '#5fe3ff' },
      temporal: { color: '#b78bff' },
      parietal: { color: '#ff6bd6' },
      occipital: { color: '#ffcf6b' },
      cerebellum: { color: '#5dffc0' },
      stem: { color: '#ff8a6b' },
    }
    const SECTOR_KEYS = Object.keys(SECTORS)
    const CLASS_KEYS = ['prompt', 'thought', 'action', 'document', 'plan', 'context']
    const CLASSES = {
      prompt: { color: '#ffb86b' },
      thought: { color: '#5fe3ff' },
      action: { color: '#ff6bd6' },
      document: { color: '#b78bff' },
      plan: { color: '#5dffc0' },
      context: { color: '#8fb8ff' },
    }

    /** Class describes what a node is; the sector keeps where it lives in the brain. */
    function classForKind(kind) {
      if (kind === 'root' || kind === 'prompt' || kind === 'instruction') return 'prompt'
      if (kind === 'reasoning' || kind === 'answer' || kind === 'attempt') return 'thought'
      if (kind === 'tool' || kind === 'agent' || kind === 'error') return 'action'
      if (kind === 'document' || kind === 'reference') return 'document'
      if (kind === 'plan' || kind === 'goal' || kind === 'deliverable') return 'plan'
      return 'context'
    }
    const ANCHORS = {
      prefrontal: { x: 0.30, y: 0.20, z: 0.56 },
      temporal: { x: 0.62, y: -0.12, z: 0.06 },
      parietal: { x: 0.40, y: 0.48, z: -0.18 },
      occipital: { x: 0.24, y: 0.02, z: -0.68 },
      cerebellum: { x: 0.00, y: -0.46, z: -0.54 },
      stem: { x: 0.00, y: -0.52, z: 0.02 },
    }
    const SECTOR_PHASE = {
      prefrontal: 0.4, temporal: 1.7, parietal: 3.1, occipital: 4.6, cerebellum: 5.5, stem: 0.9,
    }
    const FLOW_KINDS = new Set([
      'root', 'prompt', 'instruction', 'reasoning', 'answer', 'attempt', 'plan', 'goal', 'deliverable', 'error',
    ])
    const EVENT_KINDS = new Set([
      'prompt', 'reasoning', 'answer', 'tool', 'document', 'reference', 'plan', 'goal', 'deliverable',
      'agent', 'error', 'attempt',
    ])
    const CALL_FAMILIES = {
      skill: { color: '#d6b4ff' },
      mcp: { color: '#6ff0ff' },
    }

    function toolFamilyOf(name) {
      if (name === 'skill') return 'skill'
      return typeof name === 'string' && name.startsWith('mcp__') ? 'mcp' : null
    }

    function mcpNameParts(name) {
      const rest = String(name || '').slice(5)
      const split = rest.indexOf('__')
      if (split < 0) return { server: rest, tool: rest }
      return { server: rest.slice(0, split), tool: rest.slice(split + 2) }
    }

    function callFamilyColor(family) {
      return (CALL_FAMILIES[family] || {}).color || '#9fd9ff'
    }

    /** Pick at most three newly arrived events for on-brain toasts, guaranteeing a thought. */
    function pickToastEvents(fresh) {
      const committedThoughts = fresh
        .filter(event => (event.kind === 'reasoning' || event.kind === 'answer' || event.kind === 'attempt') && !event.live)
        .sort((a, b) => a.time - b.time)
      const reasoningEvents = committedThoughts.filter(event => event.kind === 'reasoning')
      const answerEvents = committedThoughts.filter(event => event.kind === 'answer')
      const actions = fresh
        .filter(event => event.kind !== 'reasoning' && event.kind !== 'answer' && event.kind !== 'attempt')
        .sort((a, b) => a.time - b.time)
      const picked = []
      if (reasoningEvents.length) picked.push(reasoningEvents[reasoningEvents.length - 1])
      if (answerEvents.length) picked.push(answerEvents[answerEvents.length - 1])
      /* Skill/MCP calls are the point of this view: keep the newest one visible. */
      const callActions = actions.filter(event => event.family === 'skill' || event.family === 'mcp')
      for (let index = callActions.length - 1; index >= 0 && picked.length < 3; index -= 1) {
        picked.push(callActions[index])
      }
      for (let index = actions.length - 1; index >= 0 && picked.length < 3; index -= 1) {
        if (!picked.includes(actions[index])) picked.push(actions[index])
      }
      if (!picked.length && committedThoughts.length) {
        picked.push(committedThoughts[committedThoughts.length - 1])
      }
      if (!picked.length) picked.push(fresh[0])
      return picked.filter(Boolean)
    }

    const TOOL_SECTORS = {
      read: 'temporal', read_image: 'occipital', glob: 'temporal', grep: 'temporal', lsp: 'temporal',
      write: 'temporal', edit: 'temporal', apply_patch: 'temporal',
      web_search: 'occipital', web_fetch: 'occipital',
      bash: 'cerebellum', job_list: 'cerebellum', job_output: 'cerebellum', job_kill: 'cerebellum',
      terminal_open: 'cerebellum', terminal_read: 'cerebellum', terminal_signal: 'cerebellum',
      terminal_close: 'cerebellum', terminal_list: 'cerebellum',
      subagent: 'cerebellum', subagent_fork: 'cerebellum', send_message: 'cerebellum',
      list_agents: 'cerebellum', wait_agent: 'cerebellum', workflow: 'cerebellum',
      todo_write: 'prefrontal', ask_user_question: 'prefrontal',
      create_goal: 'prefrontal', update_goal: 'prefrontal', get_goal: 'prefrontal',
      skill: 'temporal', present: 'occipital',
      apply_patch_text: 'temporal', office_render: 'temporal', office_convert: 'temporal',
      office_recalculate: 'temporal',
    }

    /* ------------------------------------------------------------------ *
     *  Utilities
     * ------------------------------------------------------------------ */

    const clamp = (value, min, max) => value < min ? min : value > max ? max : value

    function clip(value, max) {
      const text = typeof value === 'string' ? value : value == null ? '' : String(value)
      return text.length > max ? text.slice(0, max) + '…' : text
    }

    function firstLine(value, max) {
      return clip(String(value == null ? '' : value).replace(/\s+/g, ' ').trim(), max)
    }

    function hash(text) {
      let h = 2166136261
      const value = String(text)
      for (let i = 0; i < value.length; i += 1) {
        h ^= value.charCodeAt(i)
        h = Math.imul(h, 16777619)
      }
      return h >>> 0
    }

    function basename(path) {
      const parts = String(path || '').split(/[\\/]+/).filter(Boolean)
      return parts.length ? parts[parts.length - 1] : ''
    }

    function hostOf(url) {
      try { return new URL(String(url)).host } catch { return clip(url, 48) }
    }

    function safeJson(raw) {
      if (typeof raw !== 'string' || raw.trim() === '') return null
      try { return JSON.parse(raw) } catch { return null }
    }

    function prettyJson(raw) {
      const parsed = safeJson(raw)
      if (parsed === null) return String(raw == null ? '' : raw)
      try { return JSON.stringify(parsed, null, 2) } catch { return String(raw) }
    }

    function valueOf(value) {
      if (value == null) return ''
      if (typeof value === 'string') return value
      if (typeof value === 'number' || typeof value === 'boolean') return String(value)
      if (Array.isArray(value)) return value.map(valueOf).filter(Boolean).join(', ')
      try { return JSON.stringify(value) } catch { return String(value) }
    }

    function textOfContent(content) {
      if (!Array.isArray(content)) return ''
      const parts = []
      for (const block of content) {
        if (!block || typeof block !== 'object') continue
        if (block.type === 'text' && typeof block.text === 'string' && block.text.length) parts.push(block.text)
      }
      return parts.join('\n\n')
    }

    function reasoningOfContent(content) {
      if (!Array.isArray(content)) return []
      const parts = []
      for (const block of content) {
        if (block && block.type === 'reasoning' && typeof block.text === 'string' && block.text.trim().length) parts.push(block.text)
      }
      return parts
    }

    function blocksOfType(content, type) {
      if (!Array.isArray(content)) return []
      return content.filter(block => block && block.type === type)
    }

    function isDocPath(path) {
      return /\.(md|markdown|txt|rst|adoc|json|ya?ml|toml|csv|pdf|docx?|xlsx?|pptx?)$/i.test(String(path || ''))
    }

    function errorText(error) {
      if (!error) return ''
      if (typeof error === 'string') return error
      if (error.message) return String(error.message)
      try { return JSON.stringify(error) } catch { return String(error) }
    }

    function pickText(data) {
      if (!data || typeof data !== 'object') return ''
      const keys = ['text', 'objective', 'title', 'summary', 'reason', 'message', 'description', 'name', 'path']
      for (const key of keys) {
        const value = data[key]
        if (typeof value === 'string' && value.trim()) return value
      }
      for (const value of Object.values(data)) {
        if (typeof value === 'string' && value.trim().length > 1) return value
      }
      return ''
    }

    /* ------------------------------------------------------------------ *
     *  Dictionaries
     * ------------------------------------------------------------------ */

    const EN = {
      'pill.label': 'Neural mind',
      'pill.title': 'Neural mind — a live map of the model’s thoughts',
      'panel.title': 'Neural mind',
      'panel.subtitle': 'Live visualization of the agent’s thoughts',
      'panel.close': 'Close',
      'panel.resizeSide': 'Drag to resize the side panel',
      'session.choose': 'Active chat',
      'session.search': 'Find a chat…',
      'session.empty': 'No chats yet',
      'session.emptyHint': 'Start a conversation in the main window and the brain will come alive.',
      'session.refresh': 'Refresh',
      'session.running': 'thinking',
      'session.runningCount': '{count} thinking',
      'session.events': '{count} events',
      'session.toolCount': '{count} commands',
      'session.loadFailed': 'Could not open the chat: {message}',
      'status.thinking': 'Model is thinking…',
      'status.idle': 'Model is idle',
      'tabs.flow': 'Thoughts',
      'tabs.signals': 'Commands',
      'tabs.node': 'Node',
      'tabs.media': 'Media',
      'tabs.calls': 'Calls',
      'calls.mcpServer': 'MCP server',
      'calls.mcpTool': 'MCP tool',
      'calls.title': 'Skill and MCP calls',
      'calls.subtitle': 'Capabilities used in this chat',
      'calls.all': 'All',
      'calls.skill': 'Skills',
      'calls.mcp': 'MCP',
      'calls.empty': 'No skill or MCP calls yet',
      'calls.emptyHint': 'They appear here when the agent uses a capability.',
      'calls.emptyFiltered': 'Nothing matches',
      'calls.skillLoaded': 'Skill loaded',
      'calls.total': 'Total: {count}',
      'calls.skillHint': 'Show skill calls',
      'calls.mcpHint': 'Show MCP calls',
      'calls.skillTag': 'SKILL',
      'calls.mcpTag': 'MCP',
      'timeline.label': 'History',
      'timeline.replaying': 'Replaying history',
      'timeline.play': 'Play history',
      'timeline.pause': 'Pause',
      'timeline.restart': 'Restart',
      'timeline.back': 'Previous step',
      'timeline.forward': 'Next step',
      'timeline.speed': 'Speed',
      'timeline.loop': 'Loop',
      'timeline.live': 'Back to live',
      'timeline.ready': 'Replay this chat',
      'timeline.scrub': 'Timeline',
      'timeline.marker': 'Jump to event',
      'tabs.files': 'Files',
      'node.files': 'Files',
      'file.all': 'All',
      'file.docs': 'Docs',
      'file.data': 'Data',
      'file.code': 'Code',
      'file.text': 'Text',
      'file.empty': 'No text files yet',
      'file.emptyHint': 'Markdown, CSV, code and other text files from this chat appear here.',
      'file.open': 'Open file',
      'file.loading': 'Loading file…',
      'file.loadFailed': 'Could not load this file',
      'file.tooLarge': 'File is too large to preview ({size})',
      'file.truncated': 'Preview limited to {size}',
      'file.type': 'Type',
      'file.lines': 'Lines',
      'file.copy': 'Copy text',
      'file.copied': 'Copied',
      'file.copyFailed': 'Copy failed',
      'file.download': 'Download',
      'file.close': 'Close file',
      'file.preview': 'Preview',
      'file.table': 'Table',
      'file.moreArrived': '+{count} more files',
      'file.loadOlder': 'Show earlier files',
      'file.loadingOlder': 'Loading earlier files…',
      'file.dropped': 'Older files hidden: {count}',
      'file.droppedHint': 'The file gallery keeps the newest items.',
      'node.media': 'Media',
      'media.image': 'Image',
      'media.video': 'Video',
      'media.images': 'Images',
      'media.videos': 'Videos',
      'media.all': 'All',
      'media.empty': 'No images or videos yet',
      'media.emptyHint': 'Screenshots, renders and clips from this chat will collect here.',
      'media.untitled': 'Untitled',
      'media.more': 'Show {count} more',
      'media.loadOlder': 'Show earlier files',
      'media.loadingOlder': 'Loading earlier files…',
      'media.loading': 'Loading…',
      'media.loadFailed': 'Could not load this file',
      'media.kind': 'Type',
      'media.dimensions': 'Dimensions',
      'media.size': 'Size',
      'media.source': 'Source',
      'media.time': 'Added',
      'media.open': 'Open media',
      'media.openNode': 'Open source node',
      'media.download': 'Download',
      'media.close': 'Close media',
      'media.preview': 'Media preview',
      'media.fromPrompt': 'Your prompt',
      'media.deliverable': 'Deliverable',
      'media.dropped': 'Older media hidden: {count}',
      'media.droppedHint': 'The gallery keeps the newest items.',
      'media.moreArrived': '+{count} more arrived',
      'tabs.chats': 'Chats',
      'flow.empty': 'The reasoning stream will appear here.',
      'flow.emptyHint': 'Pick a chat where the model is working, or send it a message.',
      'flow.live': 'live thought',
      'flow.more': 'Show earlier thoughts',
      'signals.empty': 'No tool calls yet.',
      'node.empty': 'Select a node in the graph to read its context.',
      'node.args': 'Arguments',
      'node.result': 'Result',
      'node.links': 'Connections',
      'node.focus': 'Center on node',
      'node.inspect': 'Open node',
      'node.close': 'Back to stream',
      'node.noLinks': 'No connections yet',
      'node.path': 'Path',
      'node.status': 'Status',
      'node.provider': 'Provider',
      'node.model': 'Model',
      'node.reasoning': 'Reasoning effort',
      'node.contextWindow': 'Context window',
      'node.toolCount': 'Tools',
      'node.unknown': 'Unknown',
      'node.error': 'Error',
      'node.titleDerived': 'The beginning of the journal is not loaded; this task anchor uses the session title.',
      'node.kind.root': 'Task context',
      'node.kind.instruction': 'New instruction',
      'node.kind.reasoning': 'Reasoning',
      'node.kind.answer': 'Answer',
      'node.kind.context': 'Context',
      'node.kind.tool': 'Command',
      'node.kind.document': 'Document',
      'node.kind.reference': 'Reference',
      'node.kind.model': 'Model',
      'node.kind.system': 'System prompt',
      'node.kind.tools': 'Toolset',
      'node.kind.agent': 'Agent',
      'node.kind.plan': 'Plan',
      'node.kind.goal': 'Goal',
      'node.kind.deliverable': 'Deliverable',
      'node.kind.attempt': 'Retry',
      'node.kind.error': 'Failure',
      'node.kind.live': 'Live thought',
      'node.kind.prompt': 'Prompt',
      'node.kind.project': 'Project',
      'node.kind.session': 'Chat',
      'node.kind.subagent-session': 'Subagent chat',
      'node.status.running': 'running',
      'node.status.done': 'done',
      'node.status.error': 'failed',
      'sector.prefrontal': 'Prefrontal cortex',
      'sector.temporal': 'Temporal lobe',
      'sector.parietal': 'Parietal lobe',
      'sector.occipital': 'Occipital lobe',
      'sector.cerebellum': 'Cerebellum',
      'sector.stem': 'Brain stem',
      'sector.prefrontal.short': 'Prefrontal',
      'sector.temporal.short': 'Temporal',
      'sector.parietal.short': 'Parietal',
      'sector.occipital.short': 'Occipital',
      'sector.cerebellum.short': 'Cerebellum',
      'sector.stem.short': 'Stem',
      'stage.nodes': '{nodes} nodes · {edges} links',
      'stage.hint': 'drag to rotate · scroll to zoom · click a node for context',
      'stage.zoomIn': 'Zoom in',
      'stage.zoomOut': 'Zoom out',
      'stage.reset': 'Reset view',
      'stage.canvasLabel': 'Three-dimensional graph of the model’s thoughts',
      'stage.empty': 'The brain is quiet',
      'stage.emptyHint': 'This chat has no events yet.',
      'stage.legend': 'Sectors',
      'stage.partial': 'Only the loaded tail of a long journal is shown',
      'stage.capped': '{dropped} older nodes folded away',
      'stage.kicker': 'NEUROGRAPH',
      'stage.kickerProject': 'PROJECT GRAPH',
      'stage.hintProject': 'click a chat for context · open its brain from the panel',
      'stage.info': 'What each sector means',
      'stage.infoClose': 'Close sector guide',
      'stage.classes': 'Classes',
      'stage.allClasses': 'All classes',
      'stage.classesHint': 'Prompts, thoughts, actions, documents, plans and context.',
      'class.prompt': 'Prompts',
      'class.thought': 'Thoughts',
      'class.action': 'Actions',
      'class.document': 'Documents',
      'class.plan': 'Plan',
      'class.context': 'Context',
      'class.prompt.hint': 'Your own messages and instructions that steer the agent.',
      'class.thought.hint': 'Reasoning, answers, retries and the live thought.',
      'class.action.hint': 'Tool calls and agent actions.',
      'class.document.hint': 'Files, references, web reads and skills.',
      'class.plan.hint': 'Plans, goals and deliverables.',
      'class.context.hint': 'Model route, system prompt, toolset and injected context.',
      'anchor.prompt': 'My prompts',
      'stage.keyboard': 'Drag to rotate, scroll to zoom, [ and ] select a node, Enter opens it',
      'toast.from': 'from {sector}',
      'toast.dismiss': 'Dismiss',
      'loading': 'Syncing…',
      'error': 'Error: {message}',
      'retry': 'Retry',
      'model.none': 'Model not known yet',
      'model.tokens': '{n} tokens',
      'model.requests': '{n} requests',
      'anchor.task': 'Task context',
      'anchor.model': 'Model',
      'anchor.system': 'System prompt',
      'anchor.docs': 'Documents',
      'anchor.plan': 'Plan & goal',
      'time.justNow': 'just now',
      'lang.label': 'Language',
      'mode.chat': 'Chat',
      'mode.project': 'Project',
      'project.choose': 'Active project',
      'project.noProject': 'No project',
      'project.empty': 'No chats in this project yet',
      'project.emptyHint': 'Start a conversation in this folder, then return here to see its chats and connections.',
      'project.chatCount': 'Chats: {count}',
      'project.chatCountLabel': 'Chats',
      'project.partial': '{shown} of {total} chats shown',
      'project.connections': 'Connections',
      'project.open': 'Open brain',
      'project.parent': 'Parent chat',
      'project.children': 'Child chats',
      'project.related': 'Project connections',
      'project.search': 'Find a chat in the project…',
      'project.selectChat': 'Select a chat to see its context and open its brain.',
      'project.legendHub': 'Project',
      'project.legendChat': 'Chat',
      'project.legendLink': 'Fork / subagent',
      'project.legendTime': 'By time',
      'filter.placeholder': 'Filter thoughts, commands and calls…',
      'filter.clear': 'Clear filter',
      'filter.empty': 'Nothing matches the filter',
      'node.inputs': 'Inputs',
      'node.outputs': 'Outputs',
      'node.copy': 'Copy content',
      'node.copied': 'Copied',
      'node.copyFailed': 'Copy failed',
      'stage.labels': 'Labels',
      'stage.fullscreen': 'Expand',
      'stage.exitFullscreen': 'Collapse',
      'sector.prefrontal.hint': 'Reasoning, plans, goals, the task context, and what the model is trying to do.',
      'sector.temporal.hint': 'Documents, references, skills, file context, and injected workspace memory.',
      'sector.parietal.hint': 'Tool actions: reads, searches, edits, and changes to the available toolset.',
      'sector.occipital.hint': 'The outside world and the output: web search and fetch, answers, and deliverables.',
      'sector.cerebellum.hint': 'Execution: shell commands, jobs, terminals, subagents, and workflows.',
      'sector.stem.hint': 'Your prompts, the model route, system prompt, and request configuration that ground everything else.',
    }

    const RU = {
      'pill.label': 'Нейромозг',
      'pill.title': 'Нейромозг — живая карта мыслей модели',
      'panel.title': 'Нейромозг',
      'panel.subtitle': 'Живая визуализация мыслей агента',
      'panel.close': 'Закрыть',
      'panel.resizeSide': 'Потяните, чтобы изменить ширину панели',
      'session.choose': 'Активный чат',
      'session.search': 'Найти чат…',
      'session.empty': 'Пока нет чатов',
      'session.emptyHint': 'Начните разговор в основном окне — и мозг оживёт.',
      'session.refresh': 'Обновить',
      'session.running': 'думает',
      'session.runningCount': 'В работе: {count}',
      'session.events': 'Событий: {count}',
      'session.toolCount': 'Команд: {count}',
      'session.loadFailed': 'Не удалось открыть чат: {message}',
      'status.thinking': 'Модель думает…',
      'status.idle': 'Модель ожидает',
      'tabs.flow': 'Мысли',
      'tabs.signals': 'Команды',
      'tabs.node': 'Узел',
      'tabs.media': 'Медиа',
      'tabs.calls': 'Вызовы',
      'calls.mcpServer': 'MCP-сервер',
      'calls.mcpTool': 'MCP-инструмент',
      'calls.title': 'Вызовы скиллов и MCP',
      'calls.subtitle': 'Возможности, использованные в этом чате',
      'calls.all': 'Все',
      'calls.skill': 'Скиллы',
      'calls.mcp': 'MCP',
      'calls.empty': 'Пока нет вызовов скиллов и MCP',
      'calls.emptyHint': 'Они появятся здесь, когда агент использует возможность.',
      'calls.emptyFiltered': 'Ничего не найдено',
      'calls.skillLoaded': 'Скилл загружен',
      'calls.total': 'Всего: {count}',
      'calls.skillHint': 'Показать вызовы скиллов',
      'calls.mcpHint': 'Показать вызовы MCP',
      'calls.skillTag': 'СКИЛЛ',
      'calls.mcpTag': 'MCP',
      'timeline.label': 'История',
      'timeline.replaying': 'Воспроизведение истории',
      'timeline.play': 'Воспроизвести историю',
      'timeline.pause': 'Пауза',
      'timeline.restart': 'С начала',
      'timeline.back': 'Предыдущий шаг',
      'timeline.forward': 'Следующий шаг',
      'timeline.speed': 'Скорость',
      'timeline.loop': 'Повтор',
      'timeline.live': 'К текущему',
      'timeline.ready': 'Воспроизвести этот чат',
      'timeline.scrub': 'Таймлайн',
      'timeline.marker': 'Перейти к событию',
      'tabs.files': 'Файлы',
      'node.files': 'Файлы',
      'file.all': 'Все',
      'file.docs': 'Документы',
      'file.data': 'Данные',
      'file.code': 'Код',
      'file.text': 'Текст',
      'file.empty': 'Пока нет текстовых файлов',
      'file.emptyHint': 'Markdown, CSV, код и другие текстовые файлы из этого чата будут собираться здесь.',
      'file.open': 'Открыть файл',
      'file.loading': 'Загрузка файла…',
      'file.loadFailed': 'Не удалось загрузить файл',
      'file.tooLarge': 'Файл слишком большой для просмотра ({size})',
      'file.truncated': 'Просмотр ограничен {size}',
      'file.type': 'Тип',
      'file.lines': 'Строк',
      'file.copy': 'Копировать текст',
      'file.copied': 'Скопировано',
      'file.copyFailed': 'Не удалось скопировать',
      'file.download': 'Скачать',
      'file.close': 'Закрыть файл',
      'file.preview': 'Просмотр',
      'file.table': 'Таблица',
      'file.moreArrived': 'Ещё +{count} файлов',
      'file.loadOlder': 'Показать более ранние файлы',
      'file.loadingOlder': 'Загружаю более ранние файлы…',
      'file.dropped': 'Старых файлов скрыто: {count}',
      'file.droppedHint': 'В галерее остаются самые новые файлы.',
      'node.media': 'Медиа',
      'media.image': 'Изображение',
      'media.video': 'Видео',
      'media.images': 'Изображения',
      'media.videos': 'Видео',
      'media.all': 'Все',
      'media.empty': 'Пока нет изображений и видео',
      'media.emptyHint': 'Скриншоты, рендеры и клипы из этого чата будут собираться здесь.',
      'media.untitled': 'Без имени',
      'media.more': 'Показать ещё {count}',
      'media.loadOlder': 'Показать более ранние файлы',
      'media.loadingOlder': 'Загружаю более ранние файлы…',
      'media.loading': 'Загрузка…',
      'media.loadFailed': 'Не удалось загрузить файл',
      'media.kind': 'Тип',
      'media.dimensions': 'Размер кадра',
      'media.size': 'Вес',
      'media.source': 'Источник',
      'media.time': 'Добавлено',
      'media.open': 'Открыть медиа',
      'media.openNode': 'Открыть узел-источник',
      'media.download': 'Скачать',
      'media.close': 'Закрыть медиа',
      'media.preview': 'Просмотр медиа',
      'media.fromPrompt': 'Ваш промпт',
      'media.deliverable': 'Результат',
      'media.dropped': 'Старых файлов скрыто: {count}',
      'media.droppedHint': 'В галерее остаются самые новые файлы.',
      'media.moreArrived': 'Ещё +{count}',
      'tabs.chats': 'Чаты',
      'flow.empty': 'Здесь появится поток рассуждений.',
      'flow.emptyHint': 'Выберите чат, где модель уже работает, или отправьте ей сообщение.',
      'flow.live': 'живая мысль',
      'flow.more': 'Показать более ранние',
      'signals.empty': 'Пока нет вызовов инструментов.',
      'node.empty': 'Выберите узел в графе, чтобы прочитать его контекст.',
      'node.args': 'Аргументы',
      'node.result': 'Результат',
      'node.links': 'Связи',
      'node.focus': 'Показать узел',
      'node.inspect': 'Открыть узел',
      'node.close': 'Назад к потоку',
      'node.noLinks': 'Связей пока нет',
      'node.path': 'Путь',
      'node.status': 'Статус',
      'node.provider': 'Провайдер',
      'node.model': 'Модель',
      'node.reasoning': 'Уровень рассуждений',
      'node.contextWindow': 'Окно контекста',
      'node.toolCount': 'Инструментов',
      'node.unknown': 'Неизвестно',
      'node.error': 'Ошибка',
      'node.titleDerived': 'Начало журнала не загружено — здесь использован заголовок сессии.',
      'node.kind.root': 'Контекст задачи',
      'node.kind.instruction': 'Новое указание',
      'node.kind.reasoning': 'Рассуждение',
      'node.kind.answer': 'Ответ',
      'node.kind.context': 'Контекст',
      'node.kind.tool': 'Команда',
      'node.kind.document': 'Документ',
      'node.kind.reference': 'Референс',
      'node.kind.model': 'Модель',
      'node.kind.system': 'Системный промпт',
      'node.kind.tools': 'Набор инструментов',
      'node.kind.agent': 'Агент',
      'node.kind.plan': 'План',
      'node.kind.goal': 'Цель',
      'node.kind.deliverable': 'Результат',
      'node.kind.attempt': 'Повтор',
      'node.kind.error': 'Сбой',
      'node.kind.live': 'Живая мысль',
      'node.kind.prompt': 'Промпт',
      'node.kind.project': 'Проект',
      'node.kind.session': 'Чат',
      'node.kind.subagent-session': 'Чат подагента',
      'node.status.running': 'выполняется',
      'node.status.done': 'готово',
      'node.status.error': 'ошибка',
      'sector.prefrontal': 'Префронтальная кора',
      'sector.temporal': 'Височная доля',
      'sector.parietal': 'Теменная доля',
      'sector.occipital': 'Затылочная доля',
      'sector.cerebellum': 'Мозжечок',
      'sector.stem': 'Ствол мозга',
      'sector.prefrontal.short': 'Префронтальная',
      'sector.temporal.short': 'Височная',
      'sector.parietal.short': 'Теменная',
      'sector.occipital.short': 'Затылочная',
      'sector.cerebellum.short': 'Мозжечок',
      'sector.stem.short': 'Ствол',
      'stage.nodes': 'Узлов: {nodes} · связей: {edges}',
      'stage.hint': 'перетащите — поворот · колесо — масштаб · клик по узлу — контекст',
      'stage.zoomIn': 'Приблизить',
      'stage.zoomOut': 'Отдалить',
      'stage.reset': 'Сбросить ракурс',
      'stage.canvasLabel': 'Трёхмерный граф мыслей модели',
      'stage.empty': 'Мозг пока молчит',
      'stage.emptyHint': 'В этом чате ещё нет событий.',
      'stage.legend': 'Секторы',
      'stage.partial': 'Показана только загруженная часть длинного журнала',
      'stage.capped': 'Старых узлов свёрнуто: {dropped}',
      'stage.kicker': 'НЕЙРОГРАФ',
      'stage.kickerProject': 'ГРАФ ПРОЕКТА',
      'stage.hintProject': 'клик по чату — контекст · открыть его мозг можно в панели справа',
      'stage.info': 'Что означает каждый сектор',
      'stage.infoClose': 'Закрыть описание секторов',
      'stage.classes': 'Классы',
      'stage.allClasses': 'Все классы',
      'stage.classesHint': 'Промпты, мысли, действия, документы, планы и контекст.',
      'class.prompt': 'Промпты',
      'class.thought': 'Мысли',
      'class.action': 'Действия',
      'class.document': 'Документы',
      'class.plan': 'План',
      'class.context': 'Контекст',
      'class.prompt.hint': 'Ваши сообщения и указания, которые направляют агента.',
      'class.thought.hint': 'Рассуждения, ответы, повторы и живая мысль.',
      'class.action.hint': 'Вызовы инструментов и действия агентов.',
      'class.document.hint': 'Файлы, референсы, веб-страницы и скиллы.',
      'class.plan.hint': 'Планы, цели и результаты.',
      'class.context.hint': 'Маршрут модели, системный промпт, набор инструментов и внедрённый контекст.',
      'anchor.prompt': 'Мои промпты',
      'stage.keyboard': 'Перетаскивание — поворот, колесо — масштаб, [ и ] выбирают узел, Enter открывает его',
      'toast.from': 'из сектора {sector}',
      'toast.dismiss': 'Скрыть',
      'loading': 'Синхронизация…',
      'error': 'Ошибка: {message}',
      'retry': 'Повторить',
      'model.none': 'Модель пока неизвестна',
      'model.tokens': 'Токенов: {n}',
      'model.requests': 'Запросов: {n}',
      'anchor.task': 'Контекст задачи',
      'anchor.model': 'Модель',
      'anchor.system': 'Системный промпт',
      'anchor.docs': 'Документы',
      'anchor.plan': 'План и цель',
      'time.justNow': 'только что',
      'lang.label': 'Язык',
      'mode.chat': 'Чат',
      'mode.project': 'Проект',
      'project.choose': 'Активный проект',
      'project.noProject': 'Без проекта',
      'project.empty': 'В проекте пока нет чатов',
      'project.emptyHint': 'Начните разговор в этой папке и вернитесь сюда, чтобы увидеть чаты и связи.',
      'project.chatCount': 'Чатов: {count}',
      'project.chatCountLabel': 'Чатов',
      'project.partial': 'Показано: {shown} из {total}',
      'project.connections': 'Связи',
      'project.open': 'Открыть мозг',
      'project.parent': 'Родительский чат',
      'project.children': 'Дочерние чаты',
      'project.related': 'Связи проекта',
      'project.search': 'Найти чат в проекте…',
      'project.selectChat': 'Выберите чат, чтобы увидеть его контекст и открыть мозг.',
      'project.legendHub': 'Проект',
      'project.legendChat': 'Чат',
      'project.legendLink': 'Форк / подагент',
      'project.legendTime': 'По времени',
      'filter.placeholder': 'Фильтр мыслей, команд и вызовов…',
      'filter.clear': 'Очистить фильтр',
      'filter.empty': 'По фильтру ничего не найдено',
      'node.inputs': 'Входящие',
      'node.outputs': 'Исходящие',
      'node.copy': 'Копировать содержимое',
      'node.copied': 'Скопировано',
      'node.copyFailed': 'Не удалось скопировать',
      'stage.labels': 'Подписи',
      'stage.fullscreen': 'Развернуть',
      'stage.exitFullscreen': 'Свернуть',
      'sector.prefrontal.hint': 'Рассуждения, планы, цели, контекст задачи и то, чего модель добивается.',
      'sector.temporal.hint': 'Документы, референсы, скиллы, файловый контекст и внедрённая память рабочей папки.',
      'sector.parietal.hint': 'Действия инструментами: чтение, поиск, правки и изменения набора инструментов.',
      'sector.occipital.hint': 'Внешний мир и вывод: веб-поиск и загрузка страниц, ответы и результаты.',
      'sector.cerebellum.hint': 'Исполнение: shell-команды, задания, терминалы, подагенты и workflow.',
      'sector.stem.hint': 'Ваши промпты, маршрут модели, системный промпт и конфигурация запросов, на которых держится всё остальное.',
    }

    const DE = Object.assign({}, EN, {
      'pill.label': 'Neuronales Denken',
      'pill.title': 'Neuronales Denken — eine lebendige Karte der Gedanken des Modells',
      'panel.title': 'Neuronales Denken',
      'panel.subtitle': 'Live-Visualisierung der Gedanken des Agenten',
      'panel.close': 'Schließen', 'panel.resizeSide': 'Zum Ändern der Seitenbreite ziehen',
      'lang.label': 'Sprache',
      'session.choose': 'Aktiver Chat',
      'session.search': 'Chat suchen…',
      'session.empty': 'Noch keine Chats',
      'session.emptyHint': 'Starten Sie ein Gespräch im Hauptfenster – dann erwacht das Gehirn.',
      'session.refresh': 'Aktualisieren',
      'session.running': 'denkt',
      'session.runningCount': 'Denken: {count}',
      'session.events': 'Ereignisse: {count}',
      'session.toolCount': 'Befehle: {count}',
      'session.loadFailed': 'Chat konnte nicht geöffnet werden: {message}',
      'status.thinking': 'Das Modell denkt…',
      'status.idle': 'Modell ist bereit',
      'tabs.flow': 'Gedanken',
      'tabs.signals': 'Befehle',
      'tabs.node': 'Knoten',
      'flow.empty': 'Hier erscheint der Gedankenstrom.',
      'flow.emptyHint': 'Wählen Sie einen Chat, in dem das Modell arbeitet, oder senden Sie eine Nachricht.',
      'flow.live': 'lebendiger Gedanke',
      'flow.more': 'Frühere Gedanken anzeigen',
      'signals.empty': 'Noch keine Werkzeugaufrufe.',
      'node.empty': 'Wählen Sie einen Knoten im Graphen, um seinen Kontext zu lesen.',
      'node.args': 'Argumente',
      'node.result': 'Ergebnis',
      'node.links': 'Verbindungen',
      'node.focus': 'Auf Knoten zentrieren',
      'node.inspect': 'Knoten öffnen',
      'node.close': 'Zurück zum Strom',
      'node.noLinks': 'Noch keine Verbindungen',
      'node.path': 'Pfad',
      'node.provider': 'Anbieter',
      'node.model': 'Modell',
      'node.reasoning': 'Denkaufwand',
      'node.contextWindow': 'Kontextfenster',
      'node.toolCount': 'Werkzeuge',
      'node.unknown': 'Unbekannt',
      'node.error': 'Fehler',
      'node.titleDerived': 'Der Anfang des Journals ist nicht geladen; dieser Aufgabenanker nutzt den Sitzungstitel.',
      'node.inputs': 'Eingänge',
      'node.outputs': 'Ausgänge',
      'node.copy': 'Inhalt kopieren',
      'node.copied': 'Kopiert',
      'node.copyFailed': 'Kopieren fehlgeschlagen',
      'node.kind.root': 'Aufgabenkontext',
      'node.kind.instruction': 'Neue Anweisung',
      'node.kind.reasoning': 'Überlegung',
      'node.kind.answer': 'Antwort',
      'node.kind.context': 'Kontext',
      'node.kind.tool': 'Befehl',
      'node.kind.document': 'Dokument',
      'node.kind.reference': 'Referenz',
      'node.kind.model': 'Modell',
      'node.kind.system': 'System-Prompt',
      'node.kind.tools': 'Werkzeugsatz',
      'node.kind.agent': 'Agent',
      'node.kind.plan': 'Plan',
      'node.kind.goal': 'Ziel',
      'node.kind.deliverable': 'Ergebnisdatei',
      'node.kind.attempt': 'Wiederholung',
      'node.kind.error': 'Fehler',
      'node.kind.live': 'Lebendiger Gedanke',
      'node.status.running': 'läuft',
      'node.status.done': 'fertig',
      'node.status.error': 'fehlgeschlagen',
      'sector.prefrontal': 'Präfrontaler Cortex',
      'sector.temporal': 'Temporallappen',
      'sector.parietal': 'Parietallappen',
      'sector.occipital': 'Okzipitallappen',
      'sector.cerebellum': 'Kleinhirn',
      'sector.stem': 'Hirnstamm',
      'sector.prefrontal.short': 'Präfrontal',
      'sector.temporal.short': 'Temporal',
      'sector.parietal.short': 'Parietal',
      'sector.occipital.short': 'Okzipital',
      'sector.cerebellum.short': 'Kleinhirn',
      'sector.stem.short': 'Stamm',
      'sector.prefrontal.hint': 'Überlegungen, Pläne, Ziele, der Aufgabenkontext und das, was das Modell erreichen will.',
      'sector.temporal.hint': 'Dokumente, Referenzen, Skills, Dateikontext und eingespieltes Arbeitsbereichsgedächtnis.',
      'sector.parietal.hint': 'Werkzeugaktionen: Lesen, Suchen, Bearbeiten und Änderungen am Werkzeugsatz.',
      'sector.occipital.hint': 'Außenwelt und Ausgabe: Websuche und -abruf, Antworten und Ergebnisdateien.',
      'sector.cerebellum.hint': 'Ausführung: Shell-Befehle, Jobs, Terminals, Subagenten und Workflows.',
      'sector.stem.hint': 'Ihre Prompts, Modellroute, System-Prompt und Anfragekonfiguration, auf denen alles andere aufbaut.',
      'stage.nodes': 'Knoten: {nodes} · Kanten: {edges}',
      'stage.hint': 'Ziehen zum Drehen · Scrollen zum Zoomen · Knoten klicken für Kontext',
      'stage.zoomIn': 'Heranzoomen',
      'stage.zoomOut': 'Herauszoomen',
      'stage.reset': 'Ansicht zurücksetzen',
      'stage.canvasLabel': 'Dreidimensionaler Graph der Gedanken des Modells',
      'stage.empty': 'Das Gehirn ist still',
      'stage.emptyHint': 'Dieser Chat hat noch keine Ereignisse.',
      'stage.legend': 'Sektoren',
      'stage.partial': 'Nur das geladene Ende eines langen Journals wird angezeigt',
      'stage.capped': 'Ältere Knoten eingeklappt: {dropped}',
      'stage.kicker': 'NEUROGRAPH',
      'stage.keyboard': 'Ziehen zum Drehen, Scrollen zum Zoomen, [ und ] wählen einen Knoten, Enter öffnet ihn',
      'stage.labels': 'Beschriftungen',
      'stage.fullscreen': 'Erweitern',
      'stage.exitFullscreen': 'Verkleinern',
      'toast.from': 'aus {sector}',
      'toast.dismiss': 'Ausblenden',
      'loading': 'Synchronisiere…',
      'error': 'Fehler: {message}',
      'retry': 'Erneut versuchen',
      'model.none': 'Modell noch unbekannt',
      'model.tokens': 'Tokens: {n}',
      'model.requests': 'Anfragen: {n}',
      'anchor.task': 'Aufgabenkontext',
      'anchor.model': 'Modell',
      'anchor.system': 'System-Prompt',
      'anchor.docs': 'Dokumente',
      'anchor.plan': 'Plan & Ziel',
      'mode.chat': 'Chat',
      'mode.project': 'Projekt',
      'project.choose': 'Aktives Projekt',
      'project.noProject': 'Kein Projekt',
      'project.empty': 'Noch keine Chats in diesem Projekt',
      'project.emptyHint': 'Starten Sie ein Gespräch in diesem Ordner und kehren Sie hierher zurück.',
      'project.chatCount': 'Chats: {count}',
      'project.connections': 'Verbindungen',
      'project.open': 'Gehirn öffnen',
      'project.parent': 'Übergeordneter Chat',
      'project.children': 'Untergeordnete Chats',
      'project.related': 'Projektverbindungen',
      'project.search': 'Chat im Projekt suchen…',
      'project.selectChat': 'Wählen Sie einen Chat, um seinen Kontext zu sehen und sein Gehirn zu öffnen.',
      'project.legendHub': 'Projekt',
      'project.legendChat': 'Chat',
      'project.legendLink': 'Fork / Subagent',
      'project.legendTime': 'Nach Zeit',
      'filter.placeholder': 'Gedanken, Befehle und Aufrufe filtern…',
      'filter.clear': 'Filter löschen',
      'filter.empty': 'Keine Treffer für den Filter',
      'time.justNow': 'gerade eben',
    })

    const FR = Object.assign({}, EN, {
      'pill.label': 'Esprit neuronal',
      'pill.title': "Esprit neuronal — une carte vivante des pensées du modèle",
      'panel.title': 'Esprit neuronal',
      'panel.subtitle': "Visualisation vivante des pensées de l'agent",
      'panel.close': 'Fermer', 'panel.resizeSide': 'Faites glisser pour redimensionner le panneau',
      'lang.label': 'Langue',
      'session.choose': 'Chat actif',
      'session.search': 'Rechercher un chat…',
      'session.empty': 'Aucun chat pour le moment',
      'session.emptyHint': "Lancez une conversation dans la fenêtre principale : le cerveau s'animera.",
      'session.refresh': 'Actualiser',
      'session.running': 'réfléchit',
      'session.runningCount': 'En réflexion : {count}',
      'session.events': 'Événements : {count}',
      'session.toolCount': 'Commandes : {count}',
      'session.loadFailed': "Impossible d'ouvrir le chat : {message}",
      'status.thinking': 'Le modèle réfléchit…',
      'status.idle': 'Modèle au repos',
      'tabs.flow': 'Pensées',
      'tabs.signals': 'Commandes',
      'tabs.node': 'Nœud',
      'flow.empty': 'Le flux de raisonnement apparaîtra ici.',
      'flow.emptyHint': 'Choisissez un chat où le modèle travaille, ou envoyez-lui un message.',
      'flow.live': 'pensée vivante',
      'flow.more': 'Afficher les pensées précédentes',
      'signals.empty': 'Aucun appel d’outil pour le moment.',
      'node.empty': 'Sélectionnez un nœud du graphe pour lire son contexte.',
      'node.args': 'Arguments',
      'node.result': 'Résultat',
      'node.links': 'Liens',
      'node.focus': 'Centrer sur le nœud',
      'node.inspect': 'Ouvrir le nœud',
      'node.close': 'Retour au flux',
      'node.noLinks': 'Aucun lien pour le moment',
      'node.path': 'Chemin',
      'node.provider': 'Fournisseur',
      'node.model': 'Modèle',
      'node.reasoning': 'Effort de raisonnement',
      'node.contextWindow': 'Fenêtre de contexte',
      'node.toolCount': 'Outils',
      'node.unknown': 'Inconnu',
      'node.error': 'Erreur',
      'node.titleDerived': "Le début du journal n'est pas chargé ; cette ancre de tâche utilise le titre de la session.",
      'node.inputs': 'Entrées',
      'node.outputs': 'Sorties',
      'node.copy': 'Copier le contenu',
      'node.copied': 'Copié',
      'node.copyFailed': 'Échec de la copie',
      'node.kind.root': 'Contexte de la tâche',
      'node.kind.instruction': 'Nouvelle consigne',
      'node.kind.reasoning': 'Raisonnement',
      'node.kind.answer': 'Réponse',
      'node.kind.context': 'Contexte',
      'node.kind.tool': 'Commande',
      'node.kind.document': 'Document',
      'node.kind.reference': 'Référence',
      'node.kind.model': 'Modèle',
      'node.kind.system': 'Prompt système',
      'node.kind.tools': 'Jeu d’outils',
      'node.kind.agent': 'Agent',
      'node.kind.plan': 'Plan',
      'node.kind.goal': 'Objectif',
      'node.kind.deliverable': 'Livrable',
      'node.kind.attempt': 'Nouvelle tentative',
      'node.kind.error': 'Échec',
      'node.kind.live': 'Pensée vivante',
      'node.status.running': 'en cours',
      'node.status.done': 'terminé',
      'node.status.error': 'échoué',
      'sector.prefrontal': 'Cortex préfrontal',
      'sector.temporal': 'Lobe temporal',
      'sector.parietal': 'Lobe pariétal',
      'sector.occipital': 'Lobe occipital',
      'sector.cerebellum': 'Cervelet',
      'sector.stem': 'Tronc cérébral',
      'sector.prefrontal.short': 'Préfrontal',
      'sector.temporal.short': 'Temporal',
      'sector.parietal.short': 'Pariétal',
      'sector.occipital.short': 'Occipital',
      'sector.cerebellum.short': 'Cervelet',
      'sector.stem.short': 'Tronc',
      'sector.prefrontal.hint': 'Raisonnement, plans, objectifs, contexte de la tâche et ce que le modèle cherche à accomplir.',
      'sector.temporal.hint': 'Documents, références, compétences, contexte de fichiers et mémoire injectée de l’espace de travail.',
      'sector.parietal.hint': 'Actions d’outils : lectures, recherches, modifications et changements du jeu d’outils.',
      'sector.occipital.hint': 'Le monde extérieur et la sortie : recherche et récupération web, réponses et livrables.',
      'sector.cerebellum.hint': 'Exécution : commandes shell, tâches, terminaux, sous-agents et workflows.',
      'sector.stem.hint': 'Vos invites, route du modèle, prompt système et configuration des requêtes qui soutiennent tout le reste.',
      'stage.nodes': 'Nœuds : {nodes} · liens : {edges}',
      'stage.hint': 'glisser pour tourner · molette pour zoomer · cliquer un nœud pour le contexte',
      'stage.zoomIn': 'Zoomer',
      'stage.zoomOut': 'Dézoomer',
      'stage.reset': 'Réinitialiser la vue',
      'stage.canvasLabel': 'Graphe tridimensionnel des pensées du modèle',
      'stage.empty': 'Le cerveau est silencieux',
      'stage.emptyHint': "Ce chat n'a encore aucun événement.",
      'stage.legend': 'Secteurs',
      'stage.partial': 'Seule la fin chargée d’un long journal est affichée',
      'stage.capped': 'Nœuds plus anciens repliés : {dropped}',
      'stage.kicker': 'NEUROGRAPHE',
      'stage.keyboard': 'Glisser pour tourner, molette pour zoomer, [ et ] sélectionnent un nœud, Entrée l’ouvre',
      'stage.labels': 'Étiquettes',
      'stage.fullscreen': 'Agrandir',
      'stage.exitFullscreen': 'Réduire',
      'toast.from': 'depuis {sector}',
      'toast.dismiss': 'Masquer',
      'loading': 'Synchronisation…',
      'error': 'Erreur : {message}',
      'retry': 'Réessayer',
      'model.none': 'Modèle encore inconnu',
      'model.tokens': 'Tokens : {n}',
      'model.requests': 'Requêtes : {n}',
      'anchor.task': 'Contexte de la tâche',
      'anchor.model': 'Modèle',
      'anchor.system': 'Prompt système',
      'anchor.docs': 'Documents',
      'anchor.plan': 'Plan et objectif',
      'mode.chat': 'Chat',
      'mode.project': 'Projet',
      'project.choose': 'Projet actif',
      'project.noProject': 'Sans projet',
      'project.empty': 'Aucun chat dans ce projet',
      'project.emptyHint': 'Lancez une conversation dans ce dossier puis revenez ici.',
      'project.chatCount': 'Chats : {count}',
      'project.connections': 'Liens',
      'project.open': 'Ouvrir le cerveau',
      'project.parent': 'Chat parent',
      'project.children': 'Chats enfants',
      'project.related': 'Liens du projet',
      'project.search': 'Rechercher un chat dans le projet…',
      'project.selectChat': 'Sélectionnez un chat pour voir son contexte et ouvrir son cerveau.',
      'project.legendHub': 'Projet',
      'project.legendChat': 'Chat',
      'project.legendLink': 'Fork / sous-agent',
      'project.legendTime': 'Par temps',
      'filter.placeholder': 'Filtrer pensées, commandes et appels…',
      'filter.clear': 'Effacer le filtre',
      'filter.empty': 'Aucun résultat pour ce filtre',
      'time.justNow': "à l'instant",
    })

    const ES = Object.assign({}, EN, {
      'pill.label': 'Mente neuronal',
      'pill.title': 'Mente neuronal: un mapa vivo de los pensamientos del modelo',
      'panel.title': 'Mente neuronal',
      'panel.subtitle': 'Visualización viva de los pensamientos del agente',
      'panel.close': 'Cerrar', 'panel.resizeSide': 'Arrastra para cambiar el ancho del panel',
      'lang.label': 'Idioma',
      'session.choose': 'Chat activo',
      'session.search': 'Buscar un chat…',
      'session.empty': 'Aún no hay chats',
      'session.emptyHint': 'Inicia una conversación en la ventana principal y el cerebro cobrará vida.',
      'session.refresh': 'Actualizar',
      'session.running': 'pensando',
      'session.runningCount': 'Pensando: {count}',
      'session.events': 'Eventos: {count}',
      'session.toolCount': 'Comandos: {count}',
      'session.loadFailed': 'No se pudo abrir el chat: {message}',
      'status.thinking': 'El modelo está pensando…',
      'status.idle': 'El modelo está en reposo',
      'tabs.flow': 'Pensamientos',
      'tabs.signals': 'Comandos',
      'tabs.node': 'Nodo',
      'flow.empty': 'El flujo de razonamiento aparecerá aquí.',
      'flow.emptyHint': 'Elige un chat donde el modelo esté trabajando o envíale un mensaje.',
      'flow.live': 'pensamiento vivo',
      'flow.more': 'Mostrar pensamientos anteriores',
      'signals.empty': 'Aún no hay llamadas a herramientas.',
      'node.empty': 'Selecciona un nodo del grafo para leer su contexto.',
      'node.args': 'Argumentos',
      'node.result': 'Resultado',
      'node.links': 'Conexiones',
      'node.focus': 'Centrar en el nodo',
      'node.inspect': 'Abrir nodo',
      'node.close': 'Volver al flujo',
      'node.noLinks': 'Aún no hay conexiones',
      'node.path': 'Ruta',
      'node.provider': 'Proveedor',
      'node.model': 'Modelo',
      'node.reasoning': 'Esfuerzo de razonamiento',
      'node.contextWindow': 'Ventana de contexto',
      'node.toolCount': 'Herramientas',
      'node.unknown': 'Desconocido',
      'node.error': 'Error',
      'node.titleDerived': 'El inicio del diario no está cargado; este ancla de tarea usa el título de la sesión.',
      'node.inputs': 'Entradas',
      'node.outputs': 'Salidas',
      'node.copy': 'Copiar contenido',
      'node.copied': 'Copiado',
      'node.copyFailed': 'No se pudo copiar',
      'node.kind.root': 'Contexto de la tarea',
      'node.kind.instruction': 'Nueva instrucción',
      'node.kind.reasoning': 'Razonamiento',
      'node.kind.answer': 'Respuesta',
      'node.kind.context': 'Contexto',
      'node.kind.tool': 'Comando',
      'node.kind.document': 'Documento',
      'node.kind.reference': 'Referencia',
      'node.kind.model': 'Modelo',
      'node.kind.system': 'Prompt del sistema',
      'node.kind.tools': 'Conjunto de herramientas',
      'node.kind.agent': 'Agente',
      'node.kind.plan': 'Plan',
      'node.kind.goal': 'Objetivo',
      'node.kind.deliverable': 'Entregable',
      'node.kind.attempt': 'Reintento',
      'node.kind.error': 'Fallo',
      'node.kind.live': 'Pensamiento vivo',
      'node.status.running': 'en curso',
      'node.status.done': 'hecho',
      'node.status.error': 'fallido',
      'sector.prefrontal': 'Córtex prefrontal',
      'sector.temporal': 'Lóbulo temporal',
      'sector.parietal': 'Lóbulo parietal',
      'sector.occipital': 'Lóbulo occipital',
      'sector.cerebellum': 'Cerebelo',
      'sector.stem': 'Tronco encefálico',
      'sector.prefrontal.short': 'Prefrontal',
      'sector.temporal.short': 'Temporal',
      'sector.parietal.short': 'Parietal',
      'sector.occipital.short': 'Occipital',
      'sector.cerebellum.short': 'Cerebelo',
      'sector.stem.short': 'Tronco',
      'sector.prefrontal.hint': 'Razonamiento, planes, objetivos, contexto de la tarea y lo que el modelo intenta lograr.',
      'sector.temporal.hint': 'Documentos, referencias, habilidades, contexto de archivos y memoria inyectada del espacio de trabajo.',
      'sector.parietal.hint': 'Acciones de herramientas: lecturas, búsquedas, ediciones y cambios en el conjunto de herramientas.',
      'sector.occipital.hint': 'El mundo exterior y la salida: búsqueda y descarga web, respuestas y entregables.',
      'sector.cerebellum.hint': 'Ejecución: comandos de shell, tareas, terminales, subagentes y workflows.',
      'sector.stem.hint': 'Tus prompts, ruta del modelo, prompt del sistema y configuración de solicitudes que sostienen todo lo demás.',
      'stage.nodes': 'Nodos: {nodes} · enlaces: {edges}',
      'stage.hint': 'arrastra para girar · rueda para zoom · clic en un nodo para su contexto',
      'stage.zoomIn': 'Acercar',
      'stage.zoomOut': 'Alejar',
      'stage.reset': 'Restablecer vista',
      'stage.canvasLabel': 'Grafo tridimensional de los pensamientos del modelo',
      'stage.empty': 'El cerebro está en silencio',
      'stage.emptyHint': 'Este chat aún no tiene eventos.',
      'stage.legend': 'Sectores',
      'stage.partial': 'Solo se muestra la parte cargada de un diario largo',
      'stage.capped': 'Nodos antiguos plegados: {dropped}',
      'stage.kicker': 'NEURÓGRAFO',
      'stage.keyboard': 'Arrastra para girar, rueda para zoom, [ y ] seleccionan un nodo, Enter lo abre',
      'stage.labels': 'Etiquetas',
      'stage.fullscreen': 'Expandir',
      'stage.exitFullscreen': 'Contraer',
      'toast.from': 'desde {sector}',
      'toast.dismiss': 'Ocultar',
      'loading': 'Sincronizando…',
      'error': 'Error: {message}',
      'retry': 'Reintentar',
      'model.none': 'Modelo aún desconocido',
      'model.tokens': 'Tokens: {n}',
      'model.requests': 'Solicitudes: {n}',
      'anchor.task': 'Contexto de la tarea',
      'anchor.model': 'Modelo',
      'anchor.system': 'Prompt del sistema',
      'anchor.docs': 'Documentos',
      'anchor.plan': 'Plan y objetivo',
      'mode.chat': 'Chat',
      'mode.project': 'Proyecto',
      'project.choose': 'Proyecto activo',
      'project.noProject': 'Sin proyecto',
      'project.empty': 'Aún no hay chats en este proyecto',
      'project.emptyHint': 'Inicia una conversación en esta carpeta y vuelve aquí.',
      'project.chatCount': 'Chats: {count}',
      'project.connections': 'Conexiones',
      'project.open': 'Abrir cerebro',
      'project.parent': 'Chat padre',
      'project.children': 'Chats hijos',
      'project.related': 'Conexiones del proyecto',
      'project.search': 'Buscar un chat en el proyecto…',
      'project.selectChat': 'Selecciona un chat para ver su contexto y abrir su cerebro.',
      'project.legendHub': 'Proyecto',
      'project.legendChat': 'Chat',
      'project.legendLink': 'Fork / subagente',
      'project.legendTime': 'Por tiempo',
      'filter.placeholder': 'Filtrar pensamientos, comandos y llamadas…',
      'filter.clear': 'Borrar filtro',
      'filter.empty': 'Nada coincide con el filtro',
      'time.justNow': 'ahora mismo',
    })

    const IT = Object.assign({}, EN, {
      'pill.label': 'Mente neurale',
      'pill.title': 'Mente neurale: una mappa viva dei pensieri del modello',
      'panel.title': 'Mente neurale',
      'panel.subtitle': "Visualizzazione viva dei pensieri dell'agente",
      'panel.close': 'Chiudi', 'panel.resizeSide': 'Trascina per ridimensionare il pannello',
      'lang.label': 'Lingua',
      'session.choose': 'Chat attivo',
      'session.search': 'Cerca una chat…',
      'session.empty': 'Nessuna chat per ora',
      'session.emptyHint': 'Avvia una conversazione nella finestra principale e il cervello prenderà vita.',
      'session.refresh': 'Aggiorna',
      'session.running': 'sta pensando',
      'session.runningCount': 'Stanno pensando: {count}',
      'session.events': 'Eventi: {count}',
      'session.toolCount': 'Comandi: {count}',
      'session.loadFailed': 'Impossibile aprire la chat: {message}',
      'status.thinking': 'Il modello sta pensando…',
      'status.idle': 'Il modello è in attesa',
      'tabs.flow': 'Pensieri',
      'tabs.signals': 'Comandi',
      'tabs.node': 'Nodo',
      'flow.empty': 'Qui apparirà il flusso di ragionamento.',
      'flow.emptyHint': 'Scegli una chat in cui il modello sta lavorando o inviagli un messaggio.',
      'flow.live': 'pensiero vivo',
      'flow.more': 'Mostra i pensieri precedenti',
      'signals.empty': 'Nessuna chiamata agli strumenti per ora.',
      'node.empty': 'Seleziona un nodo nel grafo per leggere il suo contesto.',
      'node.args': 'Argomenti',
      'node.result': 'Risultato',
      'node.links': 'Collegamenti',
      'node.focus': 'Centra sul nodo',
      'node.inspect': 'Apri nodo',
      'node.close': 'Torna al flusso',
      'node.noLinks': 'Nessun collegamento per ora',
      'node.path': 'Percorso',
      'node.provider': 'Provider',
      'node.model': 'Modello',
      'node.reasoning': 'Sforzo di ragionamento',
      'node.contextWindow': 'Finestra di contesto',
      'node.toolCount': 'Strumenti',
      'node.unknown': 'Sconosciuto',
      'node.error': 'Errore',
      'node.titleDerived': "L'inizio del diario non è caricato; questa ancora del compito usa il titolo della sessione.",
      'node.inputs': 'Ingressi',
      'node.outputs': 'Uscite',
      'node.copy': 'Copia contenuto',
      'node.copied': 'Copiato',
      'node.copyFailed': 'Copia non riuscita',
      'node.kind.root': 'Contesto del compito',
      'node.kind.instruction': 'Nuova istruzione',
      'node.kind.reasoning': 'Ragionamento',
      'node.kind.answer': 'Risposta',
      'node.kind.context': 'Contesto',
      'node.kind.tool': 'Comando',
      'node.kind.document': 'Documento',
      'node.kind.reference': 'Riferimento',
      'node.kind.model': 'Modello',
      'node.kind.system': 'Prompt di sistema',
      'node.kind.tools': 'Set di strumenti',
      'node.kind.agent': 'Agente',
      'node.kind.plan': 'Piano',
      'node.kind.goal': 'Obiettivo',
      'node.kind.deliverable': 'Consegna',
      'node.kind.attempt': 'Nuovo tentativo',
      'node.kind.error': 'Errore',
      'node.kind.live': 'Pensiero vivo',
      'node.status.running': 'in corso',
      'node.status.done': 'fatto',
      'node.status.error': 'fallito',
      'sector.prefrontal': 'Corteccia prefrontale',
      'sector.temporal': 'Lobo temporale',
      'sector.parietal': 'Lobo parietale',
      'sector.occipital': 'Lobo occipitale',
      'sector.cerebellum': 'Cervelletto',
      'sector.stem': 'Tronco encefalico',
      'sector.prefrontal.short': 'Prefrontale',
      'sector.temporal.short': 'Temporale',
      'sector.parietal.short': 'Parietale',
      'sector.occipital.short': 'Occipitale',
      'sector.cerebellum.short': 'Cervelletto',
      'sector.stem.short': 'Tronco',
      'sector.prefrontal.hint': 'Ragionamento, piani, obiettivi, contesto del compito e ciò che il modello vuole ottenere.',
      'sector.temporal.hint': 'Documenti, riferimenti, competenze, contesto dei file e memoria iniettata dello spazio di lavoro.',
      'sector.parietal.hint': 'Azioni degli strumenti: letture, ricerche, modifiche e cambiamenti al set di strumenti.',
      'sector.occipital.hint': 'Il mondo esterno e l’output: ricerca e recupero web, risposte e consegne.',
      'sector.cerebellum.hint': 'Esecuzione: comandi shell, job, terminali, subagenti e workflow.',
      'sector.stem.hint': 'I tuoi prompt, route del modello, prompt di sistema e configurazione delle richieste su cui poggia tutto il resto.',
      'stage.nodes': 'Nodi: {nodes} · collegamenti: {edges}',
      'stage.hint': 'trascina per ruotare · rotella per lo zoom · clic su un nodo per il contesto',
      'stage.zoomIn': 'Ingrandisci',
      'stage.zoomOut': 'Riduci',
      'stage.reset': 'Reimposta vista',
      'stage.canvasLabel': 'Grafo tridimensionale dei pensieri del modello',
      'stage.empty': 'Il cervello è silenzioso',
      'stage.emptyHint': 'Questa chat non ha ancora eventi.',
      'stage.legend': 'Settori',
      'stage.partial': 'È mostrata solo la parte caricata di un diario lungo',
      'stage.capped': 'Nodi più vecchi ripiegati: {dropped}',
      'stage.kicker': 'NEUROGRAFO',
      'stage.keyboard': 'Trascina per ruotare, rotella per lo zoom, [ e ] selezionano un nodo, Invio lo apre',
      'stage.labels': 'Etichette',
      'stage.fullscreen': 'Espandi',
      'stage.exitFullscreen': 'Riduci',
      'toast.from': 'da {sector}',
      'toast.dismiss': 'Nascondi',
      'loading': 'Sincronizzazione…',
      'error': 'Errore: {message}',
      'retry': 'Riprova',
      'model.none': 'Modello ancora sconosciuto',
      'model.tokens': 'Token: {n}',
      'model.requests': 'Richieste: {n}',
      'anchor.task': 'Contesto del compito',
      'anchor.model': 'Modello',
      'anchor.system': 'Prompt di sistema',
      'anchor.docs': 'Documenti',
      'anchor.plan': 'Piano e obiettivo',
      'mode.chat': 'Chat',
      'mode.project': 'Progetto',
      'project.choose': 'Progetto attivo',
      'project.noProject': 'Senza progetto',
      'project.empty': 'Nessuna chat in questo progetto',
      'project.emptyHint': 'Avvia una conversazione in questa cartella e torna qui.',
      'project.chatCount': 'Chat: {count}',
      'project.connections': 'Collegamenti',
      'project.open': 'Apri cervello',
      'project.parent': 'Chat genitore',
      'project.children': 'Chat figli',
      'project.related': 'Collegamenti del progetto',
      'project.search': 'Cerca una chat nel progetto…',
      'project.selectChat': 'Seleziona una chat per vederne il contesto e aprire il cervello.',
      'project.legendHub': 'Progetto',
      'project.legendChat': 'Chat',
      'project.legendLink': 'Fork / subagente',
      'project.legendTime': 'Per tempo',
      'filter.placeholder': 'Filtra pensieri, comandi e chiamate…',
      'filter.clear': 'Cancella filtro',
      'filter.empty': 'Nessun risultato per il filtro',
      'time.justNow': 'proprio ora',
    })


    const PT = Object.assign({}, EN, {
      'pill.label': 'Mente neural',
      'pill.title': 'Mente neural — um mapa ao vivo dos pensamentos do modelo',
      'panel.title': 'Mente neural',
      'panel.subtitle': 'Visualização ao vivo dos pensamentos do agente',
      'panel.close': 'Fechar', 'panel.resizeSide': 'Arraste para redimensionar o painel',
      'session.choose': 'Chat ativo',
      'session.search': 'Encontrar um chat…',
      'session.empty': 'Ainda não há chats',
      'session.emptyHint': 'Inicie uma conversa na janela principal e o cérebro ganhará vida.',
      'session.refresh': 'Atualizar',
      'session.running': 'pensando',
      'session.runningCount': '{count} pensando',
      'session.events': 'Eventos: {count}',
      'session.toolCount': 'Comandos: {count}',
      'session.loadFailed': 'Não foi possível abrir o chat: {message}',
      'status.thinking': 'O modelo está pensando…',
      'status.idle': 'O modelo está ocioso',
      'tabs.flow': 'Pensamentos',
      'tabs.signals': 'Comandos',
      'tabs.node': 'Nó',
      'tabs.chats': 'Chats',
      'flow.empty': 'O fluxo de raciocínio aparecerá aqui.',
      'flow.emptyHint': 'Escolha um chat em que o modelo esteja trabalhando ou envie uma mensagem.',
      'flow.live': 'pensamento ao vivo',
      'flow.more': 'Mostrar pensamentos anteriores',
      'signals.empty': 'Ainda não há chamadas de ferramentas.',
      'node.empty': 'Selecione um nó no grafo para ler o contexto.',
      'node.args': 'Argumentos',
      'node.result': 'Resultado',
      'node.inputs': 'Entradas',
      'node.outputs': 'Saídas',
      'node.focus': 'Centrar no nó',
      'node.inspect': 'Abrir nó',
      'node.close': 'Voltar ao fluxo',
      'node.noLinks': 'Sem conexões',
      'node.path': 'Caminho',
      'node.status': 'Estado',
      'node.provider': 'Provedor',
      'node.model': 'Modelo',
      'node.reasoning': 'Esforço de raciocínio',
      'node.contextWindow': 'Janela de contexto',
      'node.toolCount': 'Ferramentas',
      'node.unknown': 'Desconhecido',
      'node.error': 'Erro',
      'node.titleDerived': 'O início do diário não foi carregado; esta âncora de tarefa usa o título do chat.',
      'node.copy': 'Copiar conteúdo',
      'node.copied': 'Copiado',
      'node.copyFailed': 'Falha ao copiar',
      'node.kind.root': 'Contexto da tarefa',
      'node.kind.instruction': 'Nova instrução',
      'node.kind.reasoning': 'Raciocínio',
      'node.kind.answer': 'Resposta',
      'node.kind.context': 'Contexto',
      'node.kind.tool': 'Comando',
      'node.kind.document': 'Documento',
      'node.kind.reference': 'Referência',
      'node.kind.model': 'Modelo',
      'node.kind.system': 'Prompt do sistema',
      'node.kind.tools': 'Ferramentas',
      'node.kind.agent': 'Agente',
      'node.kind.plan': 'Plano',
      'node.kind.goal': 'Objetivo',
      'node.kind.deliverable': 'Entregável',
      'node.kind.attempt': 'Nova tentativa',
      'node.kind.error': 'Falha',
      'node.kind.live': 'Pensamento ao vivo',
      'node.kind.project': 'Projeto',
      'node.kind.session': 'Chat',
      'node.kind.subagent-session': 'Chat do subagente',
      'node.status.running': 'em execução',
      'node.status.done': 'concluído',
      'node.status.error': 'falhou',
      'sector.prefrontal': 'Córtex pré-frontal',
      'sector.temporal': 'Lobo temporal',
      'sector.parietal': 'Lobo parietal',
      'sector.occipital': 'Lobo occipital',
      'sector.cerebellum': 'Cerebelo',
      'sector.stem': 'Tronco encefálico',
      'sector.prefrontal.short': 'Pré-frontal',
      'sector.temporal.short': 'Temporal',
      'sector.parietal.short': 'Parietal',
      'sector.occipital.short': 'Occipital',
      'sector.cerebellum.short': 'Cerebelo',
      'sector.stem.short': 'Tronco',
      'sector.prefrontal.hint': 'Raciocínio, planos, objetivos, o contexto da tarefa e o que o modelo tenta fazer.',
      'sector.temporal.hint': 'Documentos, referências, skills, contexto de arquivos e memória injetada da pasta.',
      'sector.parietal.hint': 'Ações de ferramentas: leituras, buscas, edições e mudanças no conjunto de ferramentas.',
      'sector.occipital.hint': 'O mundo exterior e a saída: busca e download na web, respostas e entregáveis.',
      'sector.cerebellum.hint': 'Execução: comandos de shell, jobs, terminais, subagentes e fluxos de trabalho.',
      'sector.stem.hint': 'Os seus prompts, a rota do modelo, o prompt do sistema e a configuração das requisições.',
      'stage.nodes': 'Nós: {nodes} · conexões: {edges}',
      'stage.hint': 'arraste para girar · role para ampliar · clique num nó para ver o contexto',
      'stage.hintProject': 'clique num chat para ver o contexto · abra o cérebro dele no painel',
      'stage.zoomIn': 'Ampliar',
      'stage.zoomOut': 'Reduzir',
      'stage.reset': 'Redefinir vista',
      'stage.canvasLabel': 'Grafo tridimensional dos pensamentos do modelo',
      'stage.empty': 'O cérebro está silencioso',
      'stage.emptyHint': 'Este chat ainda não tem eventos.',
      'stage.legend': 'Setores',
      'stage.partial': 'Apenas a parte carregada de um diário longo é mostrada',
      'stage.capped': '{dropped} nós antigos recolhidos',
      'stage.kicker': 'NEUROGRAFO',
      'stage.kickerProject': 'GRAFO DO PROJETO',
      'stage.keyboard': 'Arraste para girar, role para ampliar, [ e ] selecionam um nó, Enter o abre',
      'stage.labels': 'Rótulos',
      'stage.fullscreen': 'Expandir',
      'stage.exitFullscreen': 'Recolher',
      'stage.info': 'O que cada setor significa',
      'stage.infoClose': 'Fechar guia dos setores',
      'toast.from': 'do setor {sector}',
      'toast.dismiss': 'Dispensar',
      'loading': 'Sincronizando…',
      'error': 'Erro: {message}',
      'retry': 'Tentar de novo',
      'model.none': 'Modelo ainda desconhecido',
      'model.tokens': 'Tokens: {n}',
      'model.requests': 'Requisições: {n}',
      'anchor.task': 'Contexto da tarefa',
      'anchor.model': 'Modelo',
      'anchor.system': 'Prompt do sistema',
      'anchor.docs': 'Documentos',
      'anchor.plan': 'Plano e objetivo',
      'lang.label': 'Idioma',
      'mode.chat': 'Chat',
      'mode.project': 'Projeto',
      'project.choose': 'Projeto ativo',
      'project.noProject': 'Sem projeto',
      'project.empty': 'Ainda não há chats neste projeto',
      'project.emptyHint': 'Inicie uma conversa nesta pasta e volte aqui.',
      'project.chatCount': 'Chats: {count}',
      'project.chatCountLabel': 'Chats',
      'project.connections': 'Conexões',
      'project.open': 'Abrir cérebro',
      'project.parent': 'Chat pai',
      'project.children': 'Chats filhos',
      'project.related': 'Conexões do projeto',
      'project.search': 'Encontrar um chat no projeto…',
      'project.selectChat': 'Selecione um chat para ver o contexto e abrir o cérebro.',
      'project.legendHub': 'Projeto',
      'project.legendChat': 'Chat',
      'project.legendLink': 'Fork / subagente',
      'project.legendTime': 'Por tempo',
      'project.partial': '{shown} de {total} chats mostrados',
      'filter.placeholder': 'Filtrar pensamentos, comandos e chamadas…',
      'filter.clear': 'Limpar filtro',
      'filter.empty': 'Nada corresponde ao filtro',
      'time.justNow': 'agora mesmo',
    })

    /* ------------------------------------------------------------------ *
     *  Language registry and durable UI preference
     * ------------------------------------------------------------------ */

    const LANGS = [
      { id: 'ru', label: 'Русский' },
      { id: 'en', label: 'English' },
      { id: 'de', label: 'Deutsch' },
      { id: 'fr', label: 'Français' },
      { id: 'es', label: 'Español' },
      { id: 'it', label: 'Italiano' },
      { id: 'pt', label: 'Português' },
    ]
    Object.assign(DE, {
      'node.kind.prompt': 'Prompt', 'stage.classes': 'Klassen', 'stage.allClasses': 'Alle Klassen',
      'stage.classesHint': 'Prompts, Gedanken, Aktionen, Dokumente, Pläne und Kontext.',
      'class.prompt': 'Prompts', 'class.thought': 'Gedanken', 'class.action': 'Aktionen',
      'class.document': 'Dokumente', 'class.plan': 'Plan', 'class.context': 'Kontext',
      'class.prompt.hint': 'Ihre eigenen Nachrichten und Anweisungen, die den Agenten steuern.',
      'class.thought.hint': 'Überlegungen, Antworten, Wiederholungen und der lebendige Gedanke.',
      'class.action.hint': 'Werkzeugaufrufe und Agentenaktionen.',
      'class.document.hint': 'Dateien, Referenzen, Webseiten und Skills.',
      'class.plan.hint': 'Pläne, Ziele und Ergebnisse.',
      'class.context.hint': 'Modellroute, System-Prompt, Werkzeugsatz und eingespielter Kontext.',
      'anchor.prompt': 'Meine Prompts',
    })
    Object.assign(FR, {
      'node.kind.prompt': 'Invite', 'stage.classes': 'Classes', 'stage.allClasses': 'Toutes les classes',
      'stage.classesHint': 'Invites, réflexions, actions, documents, plans et contexte.',
      'class.prompt': 'Invites', 'class.thought': 'Réflexions', 'class.action': 'Actions',
      'class.document': 'Documents', 'class.plan': 'Plan', 'class.context': 'Contexte',
      'class.prompt.hint': 'Vos propres messages et instructions qui guident l’agent.',
      'class.thought.hint': 'Raisonnement, réponses, tentatives et pensée en direct.',
      'class.action.hint': 'Appels d’outils et actions d’agents.',
      'class.document.hint': 'Fichiers, références, pages web et compétences.',
      'class.plan.hint': 'Plans, objectifs et livrables.',
      'class.context.hint': 'Route du modèle, prompt système, outils et contexte injecté.',
      'anchor.prompt': 'Mes invites',
    })
    Object.assign(ES, {
      'node.kind.prompt': 'Prompt', 'stage.classes': 'Clases', 'stage.allClasses': 'Todas las clases',
      'stage.classesHint': 'Prompts, pensamientos, acciones, documentos, planes y contexto.',
      'class.prompt': 'Prompts', 'class.thought': 'Pensamientos', 'class.action': 'Acciones',
      'class.document': 'Documentos', 'class.plan': 'Plan', 'class.context': 'Contexto',
      'class.prompt.hint': 'Tus propios mensajes e instrucciones que guían al agente.',
      'class.thought.hint': 'Razonamiento, respuestas, reintentos y pensamiento en vivo.',
      'class.action.hint': 'Llamadas a herramientas y acciones de agentes.',
      'class.document.hint': 'Archivos, referencias, páginas web y habilidades.',
      'class.plan.hint': 'Planes, objetivos y entregables.',
      'class.context.hint': 'Ruta del modelo, prompt del sistema, herramientas y contexto inyectado.',
      'anchor.prompt': 'Mis prompts',
    })
    Object.assign(IT, {
      'node.kind.prompt': 'Prompt', 'stage.classes': 'Classi', 'stage.allClasses': 'Tutte le classi',
      'stage.classesHint': 'Prompt, pensieri, azioni, documenti, piani e contesto.',
      'class.prompt': 'Prompt', 'class.thought': 'Pensieri', 'class.action': 'Azioni',
      'class.document': 'Documenti', 'class.plan': 'Piano', 'class.context': 'Contesto',
      'class.prompt.hint': 'I tuoi messaggi e istruzioni che guidano l’agente.',
      'class.thought.hint': 'Ragionamento, risposte, tentativi e pensiero dal vivo.',
      'class.action.hint': 'Chiamate agli strumenti e azioni degli agenti.',
      'class.document.hint': 'File, riferimenti, pagine web e skill.',
      'class.plan.hint': 'Piani, obiettivi e risultati.',
      'class.context.hint': 'Rotta del modello, prompt di sistema, strumenti e contesto iniettato.',
      'anchor.prompt': 'I miei prompt',
    })
    Object.assign(PT, {
      'node.kind.prompt': 'Prompt', 'stage.classes': 'Classes', 'stage.allClasses': 'Todas as classes',
      'stage.classesHint': 'Prompts, pensamentos, ações, documentos, planos e contexto.',
      'class.prompt': 'Prompts', 'class.thought': 'Pensamentos', 'class.action': 'Ações',
      'class.document': 'Documentos', 'class.plan': 'Plano', 'class.context': 'Contexto',
      'class.prompt.hint': 'As suas mensagens e instruções que orientam o agente.',
      'class.thought.hint': 'Raciocínio, respostas, tentativas e pensamento ao vivo.',
      'class.action.hint': 'Chamadas de ferramentas e ações de agentes.',
      'class.document.hint': 'Ficheiros, referências, páginas web e skills.',
      'class.plan.hint': 'Planos, objetivos e entregáveis.',
      'class.context.hint': 'Rota do modelo, prompt de sistema, ferramentas e contexto injetado.',
      'anchor.prompt': 'Meus prompts',
    })
    Object.assign(DE, {
      'tabs.chats': 'Chats', 'node.kind.project': 'Projekt', 'node.kind.session': 'Chat',
      'node.kind.subagent-session': 'Subagent-Chat', 'project.chatCountLabel': 'Chats',
      'project.partial': '{shown} von {total} Chats angezeigt',
      'stage.kickerProject': 'PROJEKTGRAF',
      'stage.hintProject': 'Chat für Kontext anklicken · sein Gehirn im Panel öffnen',
      'stage.info': 'Was die einzelnen Sektoren bedeuten',
      'stage.infoClose': 'Sektorlegende schließen', 'node.status': 'Status',
    })
    Object.assign(FR, {
      'tabs.chats': 'Chats', 'node.kind.project': 'Projet', 'node.kind.session': 'Chat',
      'node.kind.subagent-session': 'Chat de sous-agent', 'project.chatCountLabel': 'Chats',
      'project.partial': '{shown} sur {total} chats affichés',
      'stage.kickerProject': 'GRAPHE DU PROJET',
      'stage.hintProject': 'cliquez sur un chat pour le contexte · ouvrez son cerveau dans le panneau',
      'stage.info': 'Signification de chaque secteur',
      'stage.infoClose': 'Fermer le guide des secteurs', 'node.status': 'Statut',
    })
    Object.assign(ES, {
      'tabs.chats': 'Chats', 'node.kind.project': 'Proyecto', 'node.kind.session': 'Chat',
      'node.kind.subagent-session': 'Chat de subagente', 'project.chatCountLabel': 'Chats',
      'project.partial': '{shown} de {total} chats mostrados',
      'stage.kickerProject': 'GRAFO DEL PROYECTO',
      'stage.hintProject': 'haz clic en un chat para el contexto · abre su cerebro en el panel',
      'stage.info': 'Qué significa cada sector',
      'stage.infoClose': 'Cerrar la guía de sectores', 'node.status': 'Estado',
    })
    Object.assign(IT, {
      'tabs.chats': 'Chat', 'node.kind.project': 'Progetto', 'node.kind.session': 'Chat',
      'node.kind.subagent-session': 'Chat del subagente', 'project.chatCountLabel': 'Chat',
      'project.partial': '{shown} di {total} chat mostrati',
      'stage.kickerProject': 'GRAFO DEL PROGETTO',
      'stage.hintProject': 'fai clic su una chat per il contesto · apri il suo cervello nel pannello',
      'stage.info': 'Cosa significa ogni settore',
      'stage.infoClose': 'Chiudi la guida dei settori', 'node.status': 'Stato',
    })
    Object.assign(DE, {
      'tabs.media': 'Medien', 'node.media': 'Medien', 'media.image': 'Bild', 'media.video': 'Video',
      'media.images': 'Bilder', 'media.videos': 'Videos', 'media.all': 'Alle',
      'media.empty': 'Noch keine Bilder oder Videos', 'media.emptyHint': 'Screenshots, Renderings und Clips aus diesem Chat sammeln sich hier.',
      'media.untitled': 'Ohne Titel', 'media.more': 'Noch {count} anzeigen', 'media.loadOlder': 'Frühere Dateien anzeigen', 'media.loadingOlder': 'Frühere Dateien werden geladen…', 'media.loading': 'Laden…',
      'media.loadFailed': 'Datei konnte nicht geladen werden', 'media.kind': 'Typ', 'media.dimensions': 'Abmessungen',
      'media.size': 'Größe', 'media.source': 'Quelle', 'media.time': 'Hinzugefügt', 'media.open': 'Medien öffnen',
      'media.openNode': 'Quellknoten öffnen', 'media.download': 'Herunterladen', 'media.close': 'Medien schließen',
      'media.preview': 'Medienvorschau', 'media.fromPrompt': 'Ihr Prompt', 'media.deliverable': 'Ergebnis',
      'media.dropped': 'Ältere Medien ausgeblendet: {count}', 'media.droppedHint': 'Die Galerie behält die neuesten Dateien.',
      'media.moreArrived': '+{count} weitere',
    })
    Object.assign(FR, {
      'tabs.media': 'Médias', 'node.media': 'Médias', 'media.image': 'Image', 'media.video': 'Vidéo',
      'media.images': 'Images', 'media.videos': 'Vidéos', 'media.all': 'Toutes',
      'media.empty': 'Pas encore d’images ou de vidéos', 'media.emptyHint': 'Les captures, rendus et clips de ce chat se rassemblent ici.',
      'media.untitled': 'Sans titre', 'media.more': 'Afficher {count} de plus', 'media.loadOlder': 'Afficher les fichiers plus anciens', 'media.loadingOlder': 'Chargement des fichiers plus anciens…', 'media.loading': 'Chargement…',
      'media.loadFailed': 'Impossible de charger ce fichier', 'media.kind': 'Type', 'media.dimensions': 'Dimensions',
      'media.size': 'Taille', 'media.source': 'Source', 'media.time': 'Ajouté', 'media.open': 'Ouvrir le média',
      'media.openNode': 'Ouvrir le nœud source', 'media.download': 'Télécharger', 'media.close': 'Fermer le média',
      'media.preview': 'Aperçu du média', 'media.fromPrompt': 'Votre invite', 'media.deliverable': 'Livrable',
      'media.dropped': 'Médias plus anciens masqués : {count}', 'media.droppedHint': 'La galerie conserve les fichiers les plus récents.',
      'media.moreArrived': '+{count} de plus',
    })
    Object.assign(ES, {
      'tabs.media': 'Medios', 'node.media': 'Medios', 'media.image': 'Imagen', 'media.video': 'Vídeo',
      'media.images': 'Imágenes', 'media.videos': 'Vídeos', 'media.all': 'Todos',
      'media.empty': 'Aún no hay imágenes ni vídeos', 'media.emptyHint': 'Las capturas, renders y clips de este chat se reúnen aquí.',
      'media.untitled': 'Sin título', 'media.more': 'Mostrar {count} más', 'media.loadOlder': 'Mostrar archivos anteriores', 'media.loadingOlder': 'Cargando archivos anteriores…', 'media.loading': 'Cargando…',
      'media.loadFailed': 'No se pudo cargar este archivo', 'media.kind': 'Tipo', 'media.dimensions': 'Dimensiones',
      'media.size': 'Tamaño', 'media.source': 'Origen', 'media.time': 'Añadido', 'media.open': 'Abrir medio',
      'media.openNode': 'Abrir nodo de origen', 'media.download': 'Descargar', 'media.close': 'Cerrar medio',
      'media.preview': 'Vista previa del medio', 'media.fromPrompt': 'Tu prompt', 'media.deliverable': 'Entregable',
      'media.dropped': 'Medios antiguos ocultos: {count}', 'media.droppedHint': 'La galería conserva los archivos más recientes.',
      'media.moreArrived': '+{count} más',
    })
    Object.assign(IT, {
      'tabs.media': 'Media', 'node.media': 'Media', 'media.image': 'Immagine', 'media.video': 'Video',
      'media.images': 'Immagini', 'media.videos': 'Video', 'media.all': 'Tutti',
      'media.empty': 'Ancora nessuna immagine o video', 'media.emptyHint': 'Screenshot, render e clip di questa chat si raccolgono qui.',
      'media.untitled': 'Senza titolo', 'media.more': 'Mostra altri {count}', 'media.loadOlder': 'Mostra file precedenti', 'media.loadingOlder': 'Caricamento file precedenti…', 'media.loading': 'Caricamento…',
      'media.loadFailed': 'Impossibile caricare il file', 'media.kind': 'Tipo', 'media.dimensions': 'Dimensioni',
      'media.size': 'Dimensione', 'media.source': 'Origine', 'media.time': 'Aggiunto', 'media.open': 'Apri media',
      'media.openNode': 'Apri nodo sorgente', 'media.download': 'Scarica', 'media.close': 'Chiudi media',
      'media.preview': 'Anteprima media', 'media.fromPrompt': 'Il tuo prompt', 'media.deliverable': 'Risultato',
      'media.dropped': 'Media più vecchi nascosti: {count}', 'media.droppedHint': 'La galleria mantiene i file più recenti.',
      'media.moreArrived': '+{count} altri',
    })
    Object.assign(PT, {
      'tabs.media': 'Multimédia', 'node.media': 'Multimédia', 'media.image': 'Imagem', 'media.video': 'Vídeo',
      'media.images': 'Imagens', 'media.videos': 'Vídeos', 'media.all': 'Todos',
      'media.empty': 'Ainda sem imagens ou vídeos', 'media.emptyHint': 'Capturas, renders e clips deste chat reúnem-se aqui.',
      'media.untitled': 'Sem título', 'media.more': 'Mostrar mais {count}', 'media.loadOlder': 'Mostrar ficheiros anteriores', 'media.loadingOlder': 'A carregar ficheiros anteriores…', 'media.loading': 'A carregar…',
      'media.loadFailed': 'Não foi possível carregar o ficheiro', 'media.kind': 'Tipo', 'media.dimensions': 'Dimensões',
      'media.size': 'Tamanho', 'media.source': 'Origem', 'media.time': 'Adicionado', 'media.open': 'Abrir multimédia',
      'media.openNode': 'Abrir nó de origem', 'media.download': 'Transferir', 'media.close': 'Fechar multimédia',
      'media.preview': 'Pré-visualização', 'media.fromPrompt': 'O seu prompt', 'media.deliverable': 'Entregável',
      'media.dropped': 'Multimédia antiga oculta: {count}', 'media.droppedHint': 'A galeria mantém os ficheiros mais recentes.',
      'media.moreArrived': '+{count} mais',
    })

    Object.assign(DE, {
      'tabs.calls': 'Aufrufe',
      'calls.mcpServer': 'MCP-Server',
      'calls.mcpTool': 'MCP-Werkzeug',
      'calls.title': 'Skill- und MCP-Aufrufe',
      'calls.subtitle': 'In diesem Chat genutzte Fähigkeiten',
      'calls.all': 'Alle',
      'calls.skill': 'Skills',
      'calls.mcp': 'MCP',
      'calls.empty': 'Noch keine Skill- oder MCP-Aufrufe',
      'calls.emptyHint': 'Sie erscheinen hier, wenn der Agent eine Fähigkeit nutzt.',
      'calls.emptyFiltered': 'Keine Treffer',
      'calls.skillLoaded': 'Skill geladen',
      'calls.total': 'Gesamt: {count}',
      'calls.skillHint': 'Skill-Aufrufe anzeigen',
      'calls.mcpHint': 'MCP-Aufrufe anzeigen',
      'calls.skillTag': 'SKILL',
      'calls.mcpTag': 'MCP',
    })
    Object.assign(FR, {
      'tabs.calls': 'Appels',
      'calls.mcpServer': 'Serveur MCP',
      'calls.mcpTool': 'Outil MCP',
      'calls.title': 'Appels de skills et MCP',
      'calls.subtitle': 'Capacités utilisées dans ce chat',
      'calls.all': 'Tous',
      'calls.skill': 'Skills',
      'calls.mcp': 'MCP',
      'calls.empty': 'Aucun appel de skill ou MCP',
      'calls.emptyHint': "Ils apparaîtront ici lorsque l'agent utilisera une capacité.",
      'calls.emptyFiltered': 'Aucun résultat',
      'calls.skillLoaded': 'Skill chargé',
      'calls.total': 'Total : {count}',
      'calls.skillHint': 'Afficher les appels de skills',
      'calls.mcpHint': 'Afficher les appels MCP',
      'calls.skillTag': 'SKILL',
      'calls.mcpTag': 'MCP',
    })
    Object.assign(ES, {
      'tabs.calls': 'Llamadas',
      'calls.mcpServer': 'Servidor MCP',
      'calls.mcpTool': 'Herramienta MCP',
      'calls.title': 'Llamadas de skills y MCP',
      'calls.subtitle': 'Capacidades usadas en este chat',
      'calls.all': 'Todas',
      'calls.skill': 'Skills',
      'calls.mcp': 'MCP',
      'calls.empty': 'Aún no hay llamadas de skill o MCP',
      'calls.emptyHint': 'Aparecerán aquí cuando el agente use una capacidad.',
      'calls.emptyFiltered': 'No hay resultados',
      'calls.skillLoaded': 'Skill cargado',
      'calls.total': 'Total: {count}',
      'calls.skillHint': 'Mostrar llamadas de skills',
      'calls.mcpHint': 'Mostrar llamadas MCP',
      'calls.skillTag': 'SKILL',
      'calls.mcpTag': 'MCP',
    })
    Object.assign(IT, {
      'tabs.calls': 'Chiamate',
      'calls.mcpServer': 'Server MCP',
      'calls.mcpTool': 'Strumento MCP',
      'calls.title': 'Chiamate a skill e MCP',
      'calls.subtitle': 'Capacità usate in questa chat',
      'calls.all': 'Tutte',
      'calls.skill': 'Skill',
      'calls.mcp': 'MCP',
      'calls.empty': 'Nessuna chiamata a skill o MCP',
      'calls.emptyHint': "Appariranno qui quando l'agente userà una capacità.",
      'calls.emptyFiltered': 'Nessun risultato',
      'calls.skillLoaded': 'Skill caricata',
      'calls.total': 'Totale: {count}',
      'calls.skillHint': 'Mostra le chiamate alle skill',
      'calls.mcpHint': 'Mostra le chiamate MCP',
      'calls.skillTag': 'SKILL',
      'calls.mcpTag': 'MCP',
    })
    Object.assign(PT, {
      'tabs.calls': 'Chamadas',
      'calls.mcpServer': 'Servidor MCP',
      'calls.mcpTool': 'Ferramenta MCP',
      'calls.title': 'Chamadas de skills e MCP',
      'calls.subtitle': 'Capacidades usadas neste chat',
      'calls.all': 'Todas',
      'calls.skill': 'Skills',
      'calls.mcp': 'MCP',
      'calls.empty': 'Ainda não há chamadas de skill ou MCP',
      'calls.emptyHint': 'Elas aparecerão aqui quando o agente usar uma capacidade.',
      'calls.emptyFiltered': 'Nenhum resultado',
      'calls.skillLoaded': 'Skill carregada',
      'calls.total': 'Total: {count}',
      'calls.skillHint': 'Mostrar chamadas de skills',
      'calls.mcpHint': 'Mostrar chamadas MCP',
      'calls.skillTag': 'SKILL',
      'calls.mcpTag': 'MCP',
    })

    Object.assign(DE, {
      'tabs.files': 'Dateien', 'node.files': 'Dateien', 'file.all': 'Alle', 'file.docs': 'Dokumente',
      'file.data': 'Daten', 'file.code': 'Code', 'file.text': 'Text', 'file.empty': 'Noch keine Textdateien',
      'file.emptyHint': 'Markdown, CSV, Code und andere Textdateien aus diesem Chat erscheinen hier.',
      'file.open': 'Datei öffnen', 'file.loading': 'Datei wird geladen…', 'file.loadFailed': 'Datei konnte nicht geladen werden',
      'file.tooLarge': 'Datei ist zu groß für die Vorschau ({size})', 'file.truncated': 'Vorschau auf {size} begrenzt',
      'file.type': 'Typ', 'file.lines': 'Zeilen', 'file.copy': 'Text kopieren', 'file.copied': 'Kopiert',
      'file.copyFailed': 'Kopieren fehlgeschlagen', 'file.download': 'Herunterladen', 'file.close': 'Datei schließen',
      'file.preview': 'Vorschau', 'file.table': 'Tabelle', 'file.moreArrived': '+{count} weitere Dateien',
      'file.loadOlder': 'Frühere Dateien anzeigen', 'file.loadingOlder': 'Frühere Dateien werden geladen…',
      'file.dropped': 'Ältere Dateien ausgeblendet: {count}', 'file.droppedHint': 'Die Galerie behält die neuesten Dateien.',
    })
    Object.assign(FR, {
      'tabs.files': 'Fichiers', 'node.files': 'Fichiers', 'file.all': 'Tous', 'file.docs': 'Documents',
      'file.data': 'Données', 'file.code': 'Code', 'file.text': 'Texte', 'file.empty': 'Pas encore de fichiers texte',
      'file.emptyHint': 'Les fichiers Markdown, CSV, code et autres textes de ce chat apparaissent ici.',
      'file.open': 'Ouvrir le fichier', 'file.loading': 'Chargement du fichier…', 'file.loadFailed': 'Impossible de charger ce fichier',
      'file.tooLarge': 'Fichier trop volumineux pour l’aperçu ({size})', 'file.truncated': 'Aperçu limité à {size}',
      'file.type': 'Type', 'file.lines': 'Lignes', 'file.copy': 'Copier le texte', 'file.copied': 'Copié',
      'file.copyFailed': 'Échec de la copie', 'file.download': 'Télécharger', 'file.close': 'Fermer le fichier',
      'file.preview': 'Aperçu', 'file.table': 'Tableau', 'file.moreArrived': '+{count} autres fichiers',
      'file.loadOlder': 'Afficher les fichiers plus anciens', 'file.loadingOlder': 'Chargement des fichiers plus anciens…',
      'file.dropped': 'Fichiers plus anciens masqués : {count}', 'file.droppedHint': 'La galerie conserve les fichiers les plus récents.',
    })
    Object.assign(ES, {
      'tabs.files': 'Archivos', 'node.files': 'Archivos', 'file.all': 'Todos', 'file.docs': 'Documentos',
      'file.data': 'Datos', 'file.code': 'Código', 'file.text': 'Texto', 'file.empty': 'Aún no hay archivos de texto',
      'file.emptyHint': 'Los archivos Markdown, CSV, código y otros textos de este chat aparecen aquí.',
      'file.open': 'Abrir archivo', 'file.loading': 'Cargando archivo…', 'file.loadFailed': 'No se pudo cargar este archivo',
      'file.tooLarge': 'El archivo es demasiado grande para la vista previa ({size})', 'file.truncated': 'Vista previa limitada a {size}',
      'file.type': 'Tipo', 'file.lines': 'Líneas', 'file.copy': 'Copiar texto', 'file.copied': 'Copiado',
      'file.copyFailed': 'Error al copiar', 'file.download': 'Descargar', 'file.close': 'Cerrar archivo',
      'file.preview': 'Vista previa', 'file.table': 'Tabla', 'file.moreArrived': '+{count} archivos más',
      'file.loadOlder': 'Mostrar archivos anteriores', 'file.loadingOlder': 'Cargando archivos anteriores…',
      'file.dropped': 'Archivos antiguos ocultos: {count}', 'file.droppedHint': 'La galería conserva los archivos más recientes.',
    })
    Object.assign(IT, {
      'tabs.files': 'File', 'node.files': 'File', 'file.all': 'Tutti', 'file.docs': 'Documenti',
      'file.data': 'Dati', 'file.code': 'Codice', 'file.text': 'Testo', 'file.empty': 'Ancora nessun file di testo',
      'file.emptyHint': 'File Markdown, CSV, codice e altri testi di questa chat appaiono qui.',
      'file.open': 'Apri file', 'file.loading': 'Caricamento file…', 'file.loadFailed': 'Impossibile caricare il file',
      'file.tooLarge': 'File troppo grande per l’anteprima ({size})', 'file.truncated': 'Anteprima limitata a {size}',
      'file.type': 'Tipo', 'file.lines': 'Righe', 'file.copy': 'Copia testo', 'file.copied': 'Copiato',
      'file.copyFailed': 'Copia non riuscita', 'file.download': 'Scarica', 'file.close': 'Chiudi file',
      'file.preview': 'Anteprima', 'file.table': 'Tabella', 'file.moreArrived': '+{count} altri file',
      'file.loadOlder': 'Mostra file precedenti', 'file.loadingOlder': 'Caricamento file precedenti…',
      'file.dropped': 'File più vecchi nascosti: {count}', 'file.droppedHint': 'La galleria mantiene i file più recenti.',
    })
    Object.assign(PT, {
      'tabs.files': 'Ficheiros', 'node.files': 'Ficheiros', 'file.all': 'Todos', 'file.docs': 'Documentos',
      'file.data': 'Dados', 'file.code': 'Código', 'file.text': 'Texto', 'file.empty': 'Ainda sem ficheiros de texto',
      'file.emptyHint': 'Ficheiros Markdown, CSV, código e outros textos deste chat aparecem aqui.',
      'file.open': 'Abrir ficheiro', 'file.loading': 'A carregar ficheiro…', 'file.loadFailed': 'Não foi possível carregar o ficheiro',
      'file.tooLarge': 'Ficheiro demasiado grande para pré-visualizar ({size})', 'file.truncated': 'Pré-visualização limitada a {size}',
      'file.type': 'Tipo', 'file.lines': 'Linhas', 'file.copy': 'Copiar texto', 'file.copied': 'Copiado',
      'file.copyFailed': 'Falha ao copiar', 'file.download': 'Transferir', 'file.close': 'Fechar ficheiro',
      'file.preview': 'Pré-visualização', 'file.table': 'Tabela', 'file.moreArrived': '+{count} mais ficheiros',
      'file.loadOlder': 'Mostrar ficheiros anteriores', 'file.loadingOlder': 'A carregar ficheiros anteriores…',
      'file.dropped': 'Ficheiros antigos ocultos: {count}', 'file.droppedHint': 'A galeria mantém os ficheiros mais recentes.',
    })

    Object.assign(DE, {
      'timeline.label': 'Verlauf', 'timeline.replaying': 'Verlauf wird abgespielt', 'timeline.play': 'Verlauf abspielen',
      'timeline.pause': 'Pause', 'timeline.restart': 'Neu starten', 'timeline.back': 'Vorheriger Schritt',
      'timeline.forward': 'Nächster Schritt', 'timeline.speed': 'Geschwindigkeit', 'timeline.loop': 'Schleife',
      'timeline.live': 'Zurück zu live', 'timeline.ready': 'Diesen Chat abspielen', 'timeline.scrub': 'Zeitleiste',
      'timeline.marker': 'Zum Ereignis springen',
    })
    Object.assign(FR, {
      'timeline.label': 'Historique', 'timeline.replaying': 'Lecture de l’historique', 'timeline.play': 'Lire l’historique',
      'timeline.pause': 'Pause', 'timeline.restart': 'Recommencer', 'timeline.back': 'Étape précédente',
      'timeline.forward': 'Étape suivante', 'timeline.speed': 'Vitesse', 'timeline.loop': 'Boucle',
      'timeline.live': 'Retour au direct', 'timeline.ready': 'Rejouer ce chat', 'timeline.scrub': 'Chronologie',
      'timeline.marker': 'Aller à l’événement',
    })
    Object.assign(ES, {
      'timeline.label': 'Historial', 'timeline.replaying': 'Reproduciendo historial', 'timeline.play': 'Reproducir historial',
      'timeline.pause': 'Pausa', 'timeline.restart': 'Reiniciar', 'timeline.back': 'Paso anterior',
      'timeline.forward': 'Paso siguiente', 'timeline.speed': 'Velocidad', 'timeline.loop': 'Bucle',
      'timeline.live': 'Volver al directo', 'timeline.ready': 'Reproducir este chat', 'timeline.scrub': 'Línea de tiempo',
      'timeline.marker': 'Ir al evento',
    })
    Object.assign(IT, {
      'timeline.label': 'Cronologia', 'timeline.replaying': 'Riproduzione cronologia', 'timeline.play': 'Riproduci cronologia',
      'timeline.pause': 'Pausa', 'timeline.restart': 'Ricomincia', 'timeline.back': 'Passo precedente',
      'timeline.forward': 'Passo successivo', 'timeline.speed': 'Velocità', 'timeline.loop': 'Ripeti',
      'timeline.live': 'Torna al live', 'timeline.ready': 'Riproduci questa chat', 'timeline.scrub': 'Cronologia',
      'timeline.marker': 'Vai all’evento',
    })
    Object.assign(PT, {
      'timeline.label': 'Histórico', 'timeline.replaying': 'A reproduzir histórico', 'timeline.play': 'Reproduzir histórico',
      'timeline.pause': 'Pausa', 'timeline.restart': 'Reiniciar', 'timeline.back': 'Passo anterior',
      'timeline.forward': 'Passo seguinte', 'timeline.speed': 'Velocidade', 'timeline.loop': 'Repetir',
      'timeline.live': 'Voltar ao direto', 'timeline.ready': 'Reproduzir este chat', 'timeline.scrub': 'Linha temporal',
      'timeline.marker': 'Ir para o evento',
    })

    const DICTS = { ru: RU, en: EN, de: DE, fr: FR, es: ES, it: IT, pt: PT }
    for (const dictionary of Object.values(DICTS)) {
      for (const key of Object.keys(EN)) {
        if (dictionary[key] == null) dictionary[key] = EN[key]
      }
    }
    const LANG_IDS = new Set(LANGS.map(item => item.id))
    const LANG_STORAGE_KEY = 'dsh.neural-thoughts.lang'

    function normalizeLang(value) {
      const raw = String(value || '').toLowerCase().split(/[-_]/)[0]
      return LANG_IDS.has(raw) ? raw : 'en'
    }

    function readStoredLang() {
      try {
        const value = window.localStorage.getItem(LANG_STORAGE_KEY)
        return value && LANG_IDS.has(value) ? value : null
      } catch { return null }
    }

    function storeLang(value) {
      try { window.localStorage.setItem(LANG_STORAGE_KEY, value) } catch {}
    }

    function createLanguageStore() {
      const stored = readStoredLang()
      let lang = stored || 'en'
      const listeners = new Set()
      function emit() {
        for (const listener of [...listeners]) listener()
      }
      return {
        getSnapshot() { return lang },
        subscribe(listener) {
          listeners.add(listener)
          return () => { listeners.delete(listener) }
        },
        set(value) {
          const next = normalizeLang(value)
          storeLang(next)
          if (next === lang) return
          lang = next
          emit()
        },
      }
    }

    function makeTranslate(lang) {
      const dict = DICTS[normalizeLang(lang)] || EN
      return (key, params) => {
        const template = dict[key] != null ? dict[key] : (EN[key] != null ? EN[key] : key)
        if (!params) return template
        return template.replace(/\{(\w+)\}/g, (match, name) => params[name] != null ? String(params[name]) : match)
      }
    }

    const languageStore = createLanguageStore()

    /* ------------------------------------------------------------------ *
     *  Graph model
     * ------------------------------------------------------------------ */

    function hemiOf(sector, id) {
      if (sector === 'stem' || sector === 'cerebellum') return 0
      return (hash(String(id)) % 2) === 0 ? 1 : -1
    }

    function createGraph() {
      return {
        nodes: [],
        byId: new Map(),
        edges: [],
        adj: new Map(),
        adjIn: new Map(),
        adjOut: new Map(),
        projectMode: false,
        projectMeta: null,
        callIds: new Map(),
        flowIds: [],
        rootId: null,
        liveReasonId: null,
        liveAnswerId: null,
        lastReasonId: null,
        lastActive: null,
        births: [],
        epoch: 0,
        structureDirty: true,
        windowCount: 0,
        hasMore: false,
        loadedAt: 0,
        modelInfo: null,
        tokens: null,
        toolCount: 0,
        lastChunkAt: 0,
        dropped: 0,
        cwd: '',
        media: [],
        mediaIds: new Set(),
        mediaEvents: [],
        mediaEventIds: new Set(),
        mediaDropped: 0,
        files: [],
        fileIds: new Set(),
        fileEvents: [],
        fileEventIds: new Set(),
        filesDropped: 0,
      }
    }

    function resetGraph(graph) {
      const cwd = graph.cwd || ''
      graph.nodes = []
      graph.byId.clear()
      graph.edges = []
      graph.adj = new Map()
      graph.adjIn = new Map()
      graph.adjOut = new Map()
      graph.projectMode = false
      graph.projectMeta = null
      graph.callIds.clear()
      graph.flowIds = []
      graph.rootId = null
      graph.liveReasonId = null
      graph.liveAnswerId = null
      graph.lastReasonId = null
      graph.lastActive = null
      graph.births = []
      graph.epoch += 1
      graph.structureDirty = true
      graph.windowCount = 0
      graph.hasMore = false
      graph.loadedAt = 0
      graph.modelInfo = null
      graph.tokens = null
      graph.toolCount = 0
      graph.lastChunkAt = 0
      graph.dropped = 0
      graph.cwd = cwd
      graph.media = []
      graph.mediaIds.clear()
      graph.mediaEvents = []
      graph.mediaEventIds.clear()
      graph.mediaDropped = 0
      graph.files = []
      graph.fileIds.clear()
      graph.fileEvents = []
      graph.fileEventIds.clear()
      graph.filesDropped = 0
    }

    function addNode(graph, spec) {
      const nodeClass = spec.class || classForKind(spec.kind)
      const node = {
        id: spec.id,
        kind: spec.kind,
        class: nodeClass,
        sector: spec.sector || 'parietal',
        title: spec.title || '',
        titleKey: spec.titleKey || null,
        text: clip(spec.text || '', CLIP_TEXT),
        argsRaw: clip(spec.argsRaw || '', 20000),
        result: clip(spec.result || '', 12000),
        error: spec.error || null,
        summary: spec.summary || '',
        time: spec.time || Date.now(),
        updatedAt: spec.time || Date.now(),
        weight: spec.weight || 1,
        status: spec.status || 'done',
        permanent: !!spec.permanent,
        live: !!spec.live,
        toolName: spec.toolName || null,
        family: spec.family || null,
        skillName: spec.skillName || null,
        mcpServer: spec.mcpServer || null,
        mcpTool: spec.mcpTool || null,
        hemi: spec.hemi !== undefined ? spec.hemi : hemiOf(spec.sector || 'parietal', spec.id),
        meta: Array.isArray(spec.meta) ? spec.meta : [],
        parentId: spec.parentId || null,
        synthetic: !!spec.synthetic,
        sessionId: spec.sessionId || null,
        running: !!spec.running,
        updatedAt: spec.updatedAt || spec.time || Date.now(),
        cwd: spec.cwd || '',
        origin: spec.origin || null,
        layout: spec.layout || null,
        color: spec.color || null,
        shape: spec.shape || (nodeClass === 'prompt' ? 'diamond' : null),
        ringIndex: spec.ringIndex !== undefined ? spec.ringIndex : null,
        p: { x: 0, y: 0, z: 0 },
      }
      graph.nodes.push(node)
      graph.byId.set(node.id, node)
      graph.structureDirty = true
      if (FLOW_KINDS.has(node.kind)) graph.flowIds.push(node.id)
      graph.births.push({
        id: node.id,
        parent: node.parentId,
        color: (SECTORS[node.sector] || SECTORS.parietal).color,
        at: Date.now(),
      })
      evict(graph)
      return node
    }

    /** Document and reference nodes are kept this long before older ones become evictable. */
    const KEEP_DOCUMENTS = 42

    function evictionRank(node) {
      if (node.kind === 'context') return 0
      if (node.kind === 'tool') return 1
      if (node.kind === 'reasoning' || node.kind === 'answer' || node.kind === 'instruction' || node.kind === 'attempt') return 2
      if (node.kind === 'reference' || node.kind === 'document') return 3
      return 4
    }

    function evict(graph) {
      if (graph.nodes.length <= MAX_NODES) return
      const documentCount = graph.nodes.reduce(
        (count, node) => count + (node.kind === 'document' || node.kind === 'reference' ? 1 : 0),
        0,
      )
      let target = null
      let targetRank = Infinity
      for (const node of graph.nodes) {
        if (node.permanent || node.live || node.id === graph.lastActive) continue
        const rank = evictionRank(node)
        if (rank === 3 && documentCount <= KEEP_DOCUMENTS) continue
        if (rank === 4) continue
        if (rank < targetRank || (rank === targetRank && (!target || node.time < target.time))) {
          target = node
          targetRank = rank
        }
      }
      if (!target) {
        /* Everything left is a plan/goal/deliverable anchor: retire the oldest one. */
        for (const node of graph.nodes) {
          if (node.permanent || node.live || node.id === graph.lastActive) continue
          if (!target || node.time < target.time) target = node
        }
      }
      if (!target) return
      const index = graph.nodes.indexOf(target)
      if (index < 0) return
      graph.nodes.splice(index, 1)
      graph.byId.delete(target.id)
      if (target.toolName) {
        for (const [callId, id] of graph.callIds) {
          if (id === target.id) graph.callIds.delete(callId)
        }
      }
      graph.flowIds = graph.flowIds.filter(id => id !== target.id)
      if (graph.lastActive === target.id) graph.lastActive = null
      graph.dropped = (graph.dropped || 0) + 1
      graph.structureDirty = true
    }

    function removeNode(graph, id) {
      const node = graph.byId.get(id)
      if (!node) return
      graph.byId.delete(id)
      const index = graph.nodes.indexOf(node)
      if (index >= 0) graph.nodes.splice(index, 1)
      graph.flowIds = graph.flowIds.filter(item => item !== id)
      if (graph.lastActive === id) graph.lastActive = null
      graph.structureDirty = true
    }

    /**
     * When a long journal's loaded tail does not reach the first user message,
     * the durable session title still carries the task. Anchor the brain to it
     * and disclose the substitution on the node instead of inventing history.
     */
    function ensureTaskAnchor(graph, title, sessionId) {
      if (graph.rootId) return
      const text = String(title || '').trim()
      if (!text || text === String(sessionId || '')) return
      const node = addNode(graph, {
        id: 'root-title',
        kind: 'root',
        class: 'context',
        sector: 'prefrontal',
        titleKey: 'node.kind.root',
        text,
        time: Date.now(),
        weight: 1.6,
        permanent: true,
        hemi: 0,
        status: 'done',
        synthetic: true,
      })
      /* The anchor belongs at the start of the chronology even though it was
       * derived after the loaded window was folded. */
      const index = graph.nodes.indexOf(node)
      if (index > 0) {
        graph.nodes.splice(index, 1)
        graph.nodes.unshift(node)
      }
      graph.flowIds = [node.id].concat(graph.flowIds.filter(id => id !== node.id))
      const neighbourTime = graph.nodes[1] && Number(graph.nodes[1].time)
      if (Number.isFinite(neighbourTime)) node.time = Math.min(node.time, neighbourTime - 1)
      graph.rootId = node.id
      graph.structureDirty = true
    }

    function removeLive(graph) {
      if (graph.liveReasonId) {
        removeNode(graph, graph.liveReasonId)
        graph.liveReasonId = null
      }
      if (graph.liveAnswerId) {
        removeNode(graph, graph.liveAnswerId)
        graph.liveAnswerId = null
      }
    }

    function touch(graph, node, time) {
      node.updatedAt = time || Date.now()
      graph.lastActive = node.id
    }

    function applyToolIdentity(graph, node, name) {
      const info = describeTool(name, node.argsRaw)
      if (node.sector !== info.sector) {
        node.sector = info.sector
        node.hemi = hemiOf(info.sector, node.id)
        graph.structureDirty = true
      }
      node.kind = info.kind
      node.toolName = name
      node.family = info.family || null
      node.skillName = info.skillName || null
      node.mcpServer = info.mcpServer || null
      node.mcpTool = info.mcpTool || null
      node.title = info.title
      node.titleKey = info.titleKey || null
      node.summary = info.summary
      node.meta = info.meta || []
      node.weight = info.weight
    }

    function upsertTool(graph, callId, name, argsRaw, time) {
      if (!callId) return null
      const id = 'tool:' + callId
      let node = graph.byId.get(id)
      if (!node) {
        const info = describeTool(name || 'tool', argsRaw || '')
        node = addNode(graph, {
          id,
          kind: info.kind,
          sector: info.sector,
          title: info.title,
          titleKey: info.titleKey || null,
          text: info.text || '',
          argsRaw: argsRaw || '',
          summary: info.summary,
          meta: info.meta || [],
          toolName: name || 'tool',
          family: info.family || null,
          skillName: info.skillName || null,
          mcpServer: info.mcpServer || null,
          mcpTool: info.mcpTool || null,
          status: 'running',
          weight: info.weight,
          time,
          parentId: graph.lastReasonId,
          hemi: hemiOf(info.sector, id),
        })
        graph.callIds.set(callId, id)
        graph.toolCount += 1
      } else {
        if (name) applyToolIdentity(graph, node, name)
        if (argsRaw && String(argsRaw).length > (node.argsRaw || '').length) {
          node.argsRaw = clip(argsRaw, 20000)
          const info = describeTool(node.toolName || 'tool', node.argsRaw)
          node.family = info.family || null
          node.skillName = info.skillName || null
          node.mcpServer = info.mcpServer || null
          node.mcpTool = info.mcpTool || null
          node.title = info.title
          node.titleKey = info.titleKey || null
          node.summary = info.summary
          node.meta = info.meta || []
        }
        if (node.status !== 'done' && node.status !== 'error') node.status = 'running'
        touch(graph, node, time)
      }
      return node
    }

    function upsertLiveTool(graph, callId, name, delta, time) {
      const id = 'tool:' + callId
      let node = graph.byId.get(id)
      if (!node) {
        node = upsertTool(graph, callId, name, delta || '', time)
        return node
      }
      if (name) applyToolIdentity(graph, node, name)
      if (delta) node.argsRaw = clip((node.argsRaw || '') + delta, 20000)
      const info = describeTool(node.toolName || name || 'tool', node.argsRaw)
      node.family = info.family || null
      node.skillName = info.skillName || null
      node.mcpServer = info.mcpServer || null
      node.mcpTool = info.mcpTool || null
      node.title = info.title
      node.titleKey = info.titleKey || null
      node.summary = info.summary
      node.meta = info.meta || []
      touch(graph, node, time)
      return node
    }

    function finishTool(graph, callId, isError, result, reason, time) {
      let node = graph.callIds.get(callId) ? graph.byId.get(graph.callIds.get(callId)) : null
      if (!node) node = upsertTool(graph, callId, 'tool', '', time)
      if (!node) return
      node.status = isError ? 'error' : 'done'
      if (result) node.result = clip(result, 12000)
      if (reason) node.error = reason
      touch(graph, node, time)
      return node
    }

    const MAX_MEDIA = 160
    const IMAGE_EXT = /^(png|jpe?g|webp|gif|avif|bmp|svg)$/i
    const VIDEO_EXT = /^(mp4|webm|mov|m4v|ogv|mkv)$/i
    const IMAGE_MIME = {
      png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp',
      gif: 'image/gif', avif: 'image/avif', bmp: 'image/bmp', svg: 'image/svg+xml',
    }
    const VIDEO_MIME = {
      mp4: 'video/mp4', webm: 'video/webm', mov: 'video/quicktime', m4v: 'video/x-m4v',
      ogv: 'video/ogg', mkv: 'video/x-matroska',
    }

    function mediaKindOfPath(path) {
      const ext = String(path || '').split('.').pop().toLowerCase()
      if (IMAGE_EXT.test(ext)) return { kind: 'image', mediaType: IMAGE_MIME[ext] || 'image/*' }
      if (VIDEO_EXT.test(ext)) return { kind: 'video', mediaType: VIDEO_MIME[ext] || 'video/*' }
      return null
    }

    function mediaNodeFields(node) {
      return {
        nodeId: node ? node.id : null,
        nodeKind: node ? node.kind : null,
        toolName: node ? node.toolName : null,
        sector: node ? node.sector : 'temporal',
      }
    }

    function pushMedia(graph, entry) {
      if (!entry || !entry.id) return null
      if (graph.mediaIds.has(entry.id)) return graph.media.find(item => item.id === entry.id) || null
      graph.mediaIds.add(entry.id)
      graph.media.push(entry)
      while (graph.media.length > MAX_MEDIA) {
        const dropped = graph.media.shift()
        if (dropped) graph.mediaIds.delete(dropped.id)
        graph.mediaDropped += 1
      }
      return entry
    }

    /** Every creation or access is an occurrence, even when the durable file is already known. */
    function pushMediaEvent(graph, media, time, source, index) {
      if (!media) return
      const id = ['occ', source || 'media', media.id, String(time), String(index || 0)].join(':')
      if (graph.mediaEventIds.has(id)) return
      graph.mediaEventIds.add(id)
      graph.mediaEvents.push({
        id,
        mediaId: media.id,
        time,
        source,
        name: media.name,
        kind: media.kind,
      })
      while (graph.mediaEvents.length > 240) {
        const dropped = graph.mediaEvents.shift()
        if (dropped) graph.mediaEventIds.delete(dropped.id)
      }
    }

    /** Images are durable attachment refs carried by message/tool-result content. */
    function captureImages(graph, content, node, time, source) {
      if (!Array.isArray(content)) return
      const fields = mediaNodeFields(node)
      for (let index = 0; index < content.length; index += 1) {
        const block = content[index]
        if (!block || block.type !== 'image' || !block.attachment) continue
        const attachment = block.attachment
        if (typeof attachment.attachmentId !== 'string' || !attachment.attachmentId) continue
        if (typeof attachment.mediaType !== 'string' || !attachment.mediaType.startsWith('image/')) continue
        const media = pushMedia(graph, Object.assign({
          id: 'att:' + attachment.attachmentId,
          kind: 'image',
          source,
          name: String(attachment.name || ('image-' + (index + 1))).slice(0, 140),
          mediaType: attachment.mediaType,
          bytes: Number(attachment.bytes) || 0,
          width: Number(attachment.width) || 0,
          height: Number(attachment.height) || 0,
          attachment,
          path: null,
          time,
        }, fields))
        pushMediaEvent(graph, media, time, source, index)
      }
    }

    /** Deliverable files are local paths; images and clips stay viewable through /api/file. */
    function captureLocalFiles(graph, files, node, time, source) {
      if (!Array.isArray(files)) return
      const fields = mediaNodeFields(node)
      const knownNames = new Set(graph.media.map(item => item.name))
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index]
        if (!file || typeof file.path !== 'string') continue
        const path = file.path
        const info = mediaKindOfPath(path)
        if (!info) continue
        if (!path.startsWith('/') && !/^[A-Za-z]:[\\/]/.test(path)) continue
        if (!isInsideWorkspace(path, graph.cwd)) continue
        const name = basename(path)
        /* The same render is usually also a durable image attachment; do not duplicate it. */
        if (info.kind === 'image' && knownNames.has(name)) continue
        knownNames.add(name)
        const media = pushMedia(graph, Object.assign({
          id: 'file:' + path,
          kind: info.kind,
          source,
          name,
          mediaType: info.mediaType,
          bytes: 0,
          width: 0,
          height: 0,
          attachment: null,
          path,
          description: typeof file.description === 'string' ? file.description : '',
          time,
        }, fields))
        pushMediaEvent(graph, media, time, source, index)
      }
    }

    const MAX_FILES = 120
    const TEXT_EXTENSIONS = new Set([
      'md', 'markdown', 'mdx', 'txt', 'text', 'rst', 'adoc', 'asc', 'org',
      'csv', 'tsv', 'json', 'jsonl', 'ndjson', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf',
      'log', 'xml', 'html', 'htm', 'css', 'scss', 'sass', 'less',
      'js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx', 'py', 'pyi', 'rb', 'go', 'rs', 'java', 'kt', 'kts',
      'c', 'h', 'cc', 'cpp', 'hpp', 'cs', 'php', 'swift', 'scala', 'sh', 'bash', 'zsh', 'fish',
      'ps1', 'bat', 'cmd', 'sql', 'graphql', 'gql', 'proto', 'tex', 'patch', 'diff', 'mod', 'lock',
      'sum', 'gradle', 'makefile', 'jenkinsfile', 'procfile', 'vagrantfile', 'rakefile', 'gemfile',
      'dockerfile', 'dockerignore', 'gitignore', 'gitattributes', 'editorconfig',
      'vue', 'svelte', 'astro', 'env.example',
    ])
    const TEXT_BASENAMES = new Set([
      'dockerfile', 'makefile', 'jenkinsfile', 'procfile', 'vagrantfile',
      'readme', 'license', 'licence', 'changelog', 'contributing', 'authors', 'notice',
      'go.mod', 'go.sum', '.gitignore', '.gitattributes', '.editorconfig', '.dockerignore',
      'requirements.txt', 'gemfile', 'rakefile', 'build.gradle', 'settings.gradle',
    ])
    const SECRET_BASENAMES = new Set([
      '.env', '.npmrc', '.pypirc', '.netrc', '.git-credentials', '.htpasswd', '.pgpass',
      'credentials', 'credentials.json', 'secrets', 'secrets.json', 'secrets.yaml', 'secrets.yml',
      'id_rsa', 'id_ed25519', 'id_ecdsa', 'id_dsa', 'authorized_keys',
      'kubeconfig', 'service-account.json', 'service_account.json',
    ])
    const SECRET_DIRS = ['/.ssh/', '/.aws/', '/.gnupg/', '/.dsh/', '/.git/', '/node_modules/', '/.kube/', '/.config/gcloud/']

    function normalizeFsPath(value) {
      if (typeof value !== 'string') return ''
      let path = value.trim().replace(/\\/g, '/')
      if (!path || path.includes('\0')) return ''
      const drive = /^([A-Za-z]):\//.exec(path)
      if (!path.startsWith('/') && !drive) return ''
      const prefix = drive ? drive[1].toUpperCase() + ':/' : '/'
      const rest = drive ? path.slice(3) : path.slice(1)
      const parts = []
      for (const part of rest.split('/')) {
        if (!part || part === '.') continue
        if (part === '..') {
          if (!parts.length) return ''
          parts.pop()
          continue
        }
        parts.push(part)
      }
      return prefix + parts.join('/')
    }

    /** Local files are only listed when lexically inside the Session workspace. */
    function isInsideWorkspace(path, cwd) {
      const target = normalizeFsPath(path)
      const root = normalizeFsPath(cwd)
      if (!target || !root || root === '/') return false
      const fold = /^[A-Za-z]:\//.test(root) ? value => value.toLowerCase() : value => value
      return fold(target) === fold(root) || fold(target).startsWith(fold(root) + '/')
    }

    function isSensitiveFilePath(path) {
      const lower = String(path || '').toLowerCase()
      const base = lower.split('/').pop() || ''
      if (base === '.env' || base.startsWith('.env.')) return true
      if (SECRET_BASENAMES.has(base)) return true
      if (/(^|[._-])(secret|secrets|credential|credentials|token|tokens|password|passwd|api[-_.]?key|private[-_.]?key|access[-_.]?key)([._-]|$)/.test(base)) return true
      if (/\.(pem|key|p12|pfx|jks|keystore|kdbx|gpg|asc|ppk|der)$/i.test(base)) return true
      return SECRET_DIRS.some(dir => lower.includes(dir))
    }

    function textFileInfo(path) {
      const name = basename(path)
      const lower = name.toLowerCase()
      const dot = lower.lastIndexOf('.')
      let ext = dot > 0 ? lower.slice(dot + 1) : ''
      if (!ext && TEXT_BASENAMES.has(lower)) {
        if (['readme', 'license', 'licence', 'changelog', 'contributing', 'authors', 'notice'].includes(lower)) {
          ext = 'txt'
        } else {
          ext = lower.replace(/^\./, '')
        }
      }
      if (!TEXT_EXTENSIONS.has(ext)) return null
      let group = 'text'
      if (['md', 'markdown', 'mdx', 'txt', 'text', 'rst', 'adoc', 'asc', 'org', 'tex'].includes(ext)) group = 'docs'
      else if (['csv', 'tsv', 'json', 'jsonl', 'ndjson', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf', 'log', 'xml', 'sql'].includes(ext)) group = 'data'
      else if (['html', 'htm', 'css', 'scss', 'sass', 'less', 'js', 'mjs', 'cjs', 'ts', 'tsx', 'jsx', 'py', 'pyi', 'rb', 'go', 'rs', 'java', 'kt', 'kts', 'c', 'h', 'cc', 'cpp', 'hpp', 'cs', 'php', 'swift', 'scala', 'sh', 'bash', 'zsh', 'fish', 'ps1', 'bat', 'cmd', 'graphql', 'gql', 'proto', 'vue', 'svelte', 'astro', 'patch', 'diff', 'dockerfile', 'mod'].includes(ext)) group = 'code'
      const mime = ext === 'md' || ext === 'markdown' || ext === 'mdx' ? 'text/markdown'
        : ext === 'csv' ? 'text/csv'
          : ext === 'tsv' ? 'text/tab-separated-values'
            : ext === 'json' ? 'application/json'
              : ext === 'yaml' || ext === 'yml' ? 'text/yaml'
                : ext === 'html' || ext === 'htm' ? 'text/html'
                  : 'text/plain'
      return { name, ext, group, mime }
    }

    function pushFile(graph, entry) {
      if (!entry || !entry.id) return null
      if (graph.fileIds.has(entry.id)) return graph.files.find(item => item.id === entry.id) || null
      graph.fileIds.add(entry.id)
      graph.files.push(entry)
      while (graph.files.length > MAX_FILES) {
        const dropped = graph.files.shift()
        if (dropped) graph.fileIds.delete(dropped.id)
        graph.filesDropped += 1
      }
      return entry
    }

    function pushFileEvent(graph, file, time, source, index) {
      if (!file) return
      const id = ['fileocc', source || 'file', file.id, String(time), String(index || 0)].join(':')
      if (graph.fileEventIds.has(id)) return
      graph.fileEventIds.add(id)
      graph.fileEvents.push({ id, fileId: file.id, time, source, name: file.name, ext: file.ext })
      while (graph.fileEvents.length > 240) {
        const dropped = graph.fileEvents.shift()
        if (dropped) graph.fileEventIds.delete(dropped.id)
      }
    }

    /**
     * Register one text file for the Files tab. Only paths inside the Session
     * workspace, with a text extension, and without secret-like names qualify.
     */
    function captureTextFile(graph, rawPath, node, time, source, description) {
      const path = normalizeFsPath(rawPath)
      if (!path || !isInsideWorkspace(path, graph.cwd) || isSensitiveFilePath(path)) return null
      const info = textFileInfo(path)
      if (!info) return null
      const fields = mediaNodeFields(node)
      const entry = {
        id: 'text:' + path,
        kind: 'text',
        source,
        path,
        name: info.name,
        ext: info.ext,
        group: info.group,
        mediaType: info.mime,
        description: description ? clip(description, 240) : '',
        bytes: 0,
        time,
        ...fields,
      }
      const file = pushFile(graph, entry)
      pushFileEvent(graph, file, time, source, 0)
      return file
    }

    function captureTextFiles(graph, files, node, time, source) {
      if (!Array.isArray(files)) return
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index]
        if (!file || typeof file.path !== 'string') continue
        const entry = captureTextFile(graph, file.path, node, time, source, file.description)
        if (entry) pushFileEvent(graph, entry, time, source, index + 1)
      }
    }

    function captureToolTextFiles(graph, name, argsRaw, node, time) {
      const args = safeJson(argsRaw) || {}
      const direct = args.file_path || args.path || args.file || args.notebook_path
      if (typeof direct === 'string') {
        captureTextFile(graph, direct, node, time, 'tool', String(name || 'tool'))
      }
      if (name === 'apply_patch') {
        const patch = typeof args.patch === 'string' ? args.patch : String(argsRaw || '')
        const lines = patch.split('\n')
        for (const line of lines) {
          const match = /^\*\*\* (?:Add|Update) File: (.+)$/.exec(line.trim())
          if (match) captureTextFile(graph, match[1].trim(), node, time, 'tool', 'apply_patch')
        }
      }
    }

    function setModel(graph, provider, model, extra) {
      const options = extra || {}
      if (!provider && !model && options.contextWindow === undefined && options.toolCount === undefined) return
      const previous = graph.modelInfo || {}
      const next = {
        provider: provider || previous.provider || '',
        model: model || previous.model || '',
        reasoningEffort: options.reasoningEffort !== undefined ? options.reasoningEffort : previous.reasoningEffort,
        contextWindow: options.contextWindow !== undefined ? options.contextWindow : previous.contextWindow,
        toolCount: options.toolCount !== undefined ? options.toolCount : previous.toolCount,
        requests: (previous.requests || 0) + (options.countRequest ? 1 : 0),
      }
      graph.modelInfo = next
      const title = [next.provider, next.model].filter(Boolean).join(' · ')
      let node = graph.byId.get('model')
      if (!node) {
        node = addNode(graph, {
          id: 'model', kind: 'model', sector: 'stem', title, permanent: true, weight: 1.4, hemi: 0, time: Date.now(),
        })
      } else {
        node.title = title
        node.updatedAt = Date.now()
      }
      const meta = []
      if (next.provider) meta.push({ key: 'node.provider', value: next.provider })
      if (next.model) meta.push({ key: 'node.model', value: next.model })
      if (next.reasoningEffort) meta.push({ key: 'node.reasoning', value: String(next.reasoningEffort) })
      if (next.contextWindow) meta.push({ key: 'node.contextWindow', value: formatTokens(next.contextWindow) })
      if (next.toolCount !== undefined) meta.push({ key: 'node.toolCount', value: String(next.toolCount) })
      node.meta = meta
    }

    function formatTokens(value) {
      const number = Number(value)
      if (!Number.isFinite(number)) return String(value)
      if (number >= 1000000) return (number / 1000000).toFixed(1).replace(/\.0$/, '') + 'M'
      if (number >= 1000) return (number / 1000).toFixed(1).replace(/\.0$/, '') + 'k'
      return String(Math.round(number))
    }

    function tokensOf(usage) {
      if (!usage || typeof usage !== 'object') return 0
      if (Number.isFinite(Number(usage.totalTokens)) && Number(usage.totalTokens) > 0) return Number(usage.totalTokens)
      const prompt = Number(usage.inputTokens || 0) + Number(usage.cacheReadTokens || 0) + Number(usage.cacheWriteTokens || 0)
      return prompt + Number(usage.outputTokens || 0)
    }

    /* ------------------------------------------------------------------ *
     *  Tool classification
     * ------------------------------------------------------------------ */

    function sectorForTool(name) {
      if (!name) return 'parietal'
      if (toolFamilyOf(name) === 'mcp') return 'occipital'
      if (TOOL_SECTORS[name]) return TOOL_SECTORS[name]
      if (/^(read|glob|grep|lsp)/.test(name)) return 'temporal'
      if (/^(write|edit|apply_patch)/.test(name)) return 'temporal'
      if (/^web/.test(name)) return 'occipital'
      if (/^(job|terminal|subagent|send_message|wait_agent|workflow)/.test(name)) return 'cerebellum'
      if (/^(todo|goal|plan|ask_user)/.test(name)) return 'prefrontal'
      return 'parietal'
    }

    function kindForTool(name, path) {
      if (!name) return 'tool'
      if (name === 'todo_write') return 'plan'
      if (/^goal/.test(name) || name === 'create_goal' || name === 'update_goal' || name === 'get_goal') return 'goal'
      if (name === 'subagent' || name === 'subagent_fork' || name === 'send_message' || name === 'wait_agent' || name === 'list_agents') return 'agent'
      if (name === 'present') return 'deliverable'
      if (name === 'write' || name === 'edit' || name === 'apply_patch') return 'document'
      if (name === 'read' || name === 'read_image' || name === 'glob' || name === 'grep') return isDocPath(path) ? 'document' : 'reference'
      if (name === 'skill' || name === 'web_search' || name === 'web_fetch') return 'reference'
      return 'tool'
    }

    function describeTool(name, argsRaw) {
      const tool = name || 'tool'
      const args = safeJson(argsRaw) || {}
      const sector = sectorForTool(tool)
      const path = typeof args.file_path === 'string' ? args.file_path
        : typeof args.path === 'string' ? args.path
          : typeof args.file === 'string' ? args.file
            : typeof args.notebook_path === 'string' ? args.notebook_path : ''
      const base = path ? basename(path) : ''
      const kind = kindForTool(tool, path)
      const meta = path ? [{ key: 'node.path', value: path }] : []

      if (tool === 'bash') {
        return {
          title: 'bash',
          summary: firstLine(args.command || argsRaw, 160),
          kind: 'tool',
          sector,
          weight: 1.0,
          meta,
        }
      }
      if (tool === 'write' || tool === 'edit' || tool === 'apply_patch') {
        return {
          title: base || tool,
          summary: path || firstLine(argsRaw, 120),
          kind: kind === 'document' ? 'document' : 'tool',
          sector,
          weight: tool === 'write' ? 1.25 : 1.15,
          meta,
        }
      }
      if (tool === 'read' || tool === 'read_image') {
        return {
          title: base || tool,
          summary: path,
          kind,
          sector,
          weight: 1.0,
          meta,
        }
      }
      if (tool === 'glob' || tool === 'grep') {
        return {
          title: tool,
          summary: String(args.pattern || args.query || firstLine(argsRaw, 120)),
          kind: 'reference',
          sector,
          weight: 0.85,
          meta,
        }
      }
      if (tool === 'web_search') {
        return {
          title: 'web_search',
          summary: firstLine(valueOf(args.queries || args.query || argsRaw), 150),
          kind: 'reference',
          sector,
          weight: 1.0,
          meta,
        }
      }
      if (tool === 'web_fetch') {
        return {
          title: hostOf(args.url) || 'web_fetch',
          summary: String(args.url || ''),
          kind: 'reference',
          sector,
          weight: 1.0,
          meta,
        }
      }
      if (tool === 'skill') {
        const skill = args.name || args.skill || base || 'skill'
        return {
          title: String(skill),
          summary: firstLine(args.args || '', 120),
          kind: 'reference',
          sector,
          weight: 1.1,
          family: 'skill',
          skillName: String(skill),
          meta,
        }
      }
      if (toolFamilyOf(tool) === 'mcp') {
        const parts = mcpNameParts(tool)
        return {
          title: parts.tool || tool,
          summary: firstLine(argsRaw, 160),
          kind: 'tool',
          sector,
          weight: 1.15,
          family: 'mcp',
          mcpServer: parts.server,
          mcpTool: parts.tool,
          meta,
        }
      }
      if (tool === 'todo_write') {
        const todos = Array.isArray(args.todos) ? args.todos : []
        const lines = todos.map(item => typeof item === 'string' ? item : (item && item.content) || '').filter(Boolean)
        return {
          title: 'todo_write',
          summary: lines.length ? firstLine(lines[0], 140) : '',
          kind: 'plan',
          sector,
          weight: 1.2,
          text: lines.join('\n'),
          meta,
        }
      }
      if (/^goal/.test(tool) || tool === 'create_goal' || tool === 'update_goal' || tool === 'get_goal') {
        return {
          title: tool,
          summary: firstLine(args.objective || args.text || argsRaw, 160),
          kind: 'goal',
          sector,
          weight: 1.2,
          text: String(args.objective || args.text || ''),
          meta,
        }
      }
      if (tool === 'subagent' || tool === 'subagent_fork') {
        return {
          title: firstLine(args.description || args.name || tool, 40),
          summary: firstLine(args.prompt || argsRaw, 160),
          kind: 'agent',
          sector,
          weight: 1.2,
          meta,
        }
      }
      if (tool === 'send_message' || tool === 'wait_agent' || tool === 'list_agents') {
        return {
          title: tool,
          summary: firstLine(args.agent || args.message || argsRaw, 150),
          kind: 'agent',
          sector,
          weight: 0.95,
          meta,
        }
      }
      if (tool === 'present') {
        const files = Array.isArray(args.files) ? args.files : []
        const first = files[0] || {}
        const firstPath = typeof first.path === 'string' ? first.path : typeof args.path === 'string' ? args.path : ''
        return {
          title: basename(firstPath) || 'present',
          summary: files.length > 1 ? firstLine(files.map(file => basename(file.path)).join(', '), 140) : firstPath,
          kind: 'deliverable',
          sector,
          weight: 1.25,
          meta: firstPath ? [{ key: 'node.path', value: firstPath }] : [],
        }
      }
      if (tool === 'ask_user_question') {
        return {
          title: 'ask_user_question',
          summary: firstLine(args.questions ? valueOf(args.questions) : argsRaw, 150),
          kind: 'tool',
          sector,
          weight: 1.05,
          meta,
        }
      }
      if (tool === 'apply_patch' && args.patch) {
        return { title: base || 'apply_patch', summary: firstLine(args.patch, 150), kind: 'document', sector, weight: 1.2, meta }
      }
      return {
        title: tool,
        summary: briefArgs(args, argsRaw),
        kind,
        sector,
        weight: 0.9,
        meta,
      }
    }

    function briefArgs(args, raw) {
      for (const key of ['description', 'command', 'query', 'pattern', 'prompt', 'url', 'name', 'text', 'objective']) {
        const value = args[key]
        if (typeof value === 'string' && value.trim()) return firstLine(value, 150)
      }
      if (Array.isArray(args.queries)) return firstLine(valueOf(args.queries), 150)
      return firstLine(raw, 120)
    }

    /* ------------------------------------------------------------------ *
     *  Folding the Session journal into the graph
     * ------------------------------------------------------------------ */

    function foldWindow(graph, window) {
      resetGraph(graph)
      const entries = window && Array.isArray(window.entries) ? window.entries : []
      for (const entry of entries) foldEntry(graph, entry)
      graph.windowCount = entries.length
      graph.hasMore = !!(window && window.hasMore)
      graph.loadedAt = Date.now()
      graph.structureDirty = true
    }

    function foldEntry(graph, entry) {
      if (!entry || !entry.event) return
      if (entry.type === 'transient') foldChunk(graph, entry.event)
      else foldEvent(graph, entry.event)
    }

    function foldChunk(graph, event) {
      const data = event.data || {}
      const chunk = data.chunk || {}
      const time = Number(event.time) || Date.now()
      graph.lastChunkAt = Date.now()
      if (chunk.type === 'reasoning-delta') {
        let node = graph.liveReasonId ? graph.byId.get(graph.liveReasonId) : null
        if (!node) {
          node = addNode(graph, {
            id: 'live-reasoning',
            kind: 'reasoning',
            sector: 'prefrontal',
            titleKey: 'node.kind.live',
            text: '',
            time,
            weight: 1.15,
            live: true,
            status: 'running',
          })
          graph.liveReasonId = node.id
        }
        node.text = clip(node.text + (chunk.text || ''), CLIP_TEXT)
        touch(graph, node, time)
        return
      }
      if (chunk.type === 'text-delta') {
        let node = graph.liveAnswerId ? graph.byId.get(graph.liveAnswerId) : null
        if (!node) {
          node = addNode(graph, {
            id: 'live-answer',
            kind: 'answer',
            sector: 'occipital',
            titleKey: 'node.kind.answer',
            text: '',
            time,
            weight: 1.1,
            live: true,
            status: 'running',
          })
          graph.liveAnswerId = node.id
        }
        node.text = clip(node.text + (chunk.text || ''), CLIP_TEXT)
        touch(graph, node, time)
        return
      }
      if (chunk.type === 'tool-call-delta') {
        if (chunk.id) upsertLiveTool(graph, chunk.id, chunk.name, chunk.argumentsDelta || '', time)
        return
      }
      if (chunk.type === 'block-end') {
        const block = chunk.block
        if (block && block.type === 'tool-call') {
          upsertTool(graph, block.id, block.name, block.arguments || '', time)
        } else if (block && block.type === 'reasoning' && graph.liveReasonId) {
          const node = graph.byId.get(graph.liveReasonId)
          if (node) node.text = clip(block.text || node.text, CLIP_TEXT)
        }
        return
      }
      if (chunk.type === 'usage' && chunk.usage) graph.tokens = chunk.usage
    }

    function foldEvent(graph, event) {
      const data = event.data || {}
      const time = Number(event.time) || Date.now()

      switch (event.type) {
        case 'turn/start': {
          graph.turn = data.turn
          return
        }
        case 'step/start': {
          graph.step = data.step
          return
        }
        case 'user/message': {
          const content = data.content || []
          const source = data.source || {}
          const text = textOfContent(content)
          if (!text && !(Array.isArray(content) && content.length)) return
          if (source.kind === 'user') {
            if (!graph.rootId) {
              const node = addNode(graph, {
                id: 'root',
                kind: 'root',
                sector: 'stem',
                titleKey: 'node.kind.root',
                text,
                time,
                weight: 1.35,
                permanent: true,
                hemi: 0,
                status: 'done',
              })
              graph.rootId = node.id
              graph.lastActive = node.id
              captureImages(graph, content, node, time, 'prompt')
            } else {
              const node = addNode(graph, {
                id: 'user:' + event.seq,
                kind: 'prompt',
                sector: 'stem',
                title: firstLine(text, 44) || 'prompt',
                text,
                time,
                weight: 1.05,
                status: 'done',
              })
              graph.lastActive = node.id
              captureImages(graph, content, node, time, 'prompt')
            }
          } else {
            const form = source.form || ''
            const titleKey = form === 'instructions' ? 'node.kind.context'
              : form === 'catalog' ? 'node.kind.tools'
                : form === 'notice' ? 'node.kind.context'
                  : form === 'relay' ? 'node.kind.agent'
                    : form === 'recall' ? 'node.kind.reference' : 'node.kind.context'
            addNode(graph, {
              id: 'ctx:' + event.seq,
              kind: 'context',
              sector: 'temporal',
              titleKey,
              text,
              time,
              weight: 0.7,
              status: 'done',
            })
          }
          return
        }
        case 'system/message': {
          const text = textOfContent(data.message && data.message.content)
          let node = graph.byId.get('system')
          if (!node) {
            node = addNode(graph, {
              id: 'system',
              kind: 'system',
              sector: 'stem',
              titleKey: 'node.kind.system',
              text,
              time,
              weight: 1.25,
              permanent: true,
              hemi: 0,
              status: 'done',
            })
          } else {
            node.text = clip(text, CLIP_TEXT)
            node.updatedAt = time
          }
          return
        }
        case 'developer/message': {
          const additions = blocksOfType(data.message && data.message.content, 'tool-addition')
            .map(block => block.toolName).filter(Boolean)
          const removals = blocksOfType(data.message && data.message.content, 'tool-removal')
            .map(block => block.toolName).filter(Boolean)
          let node = graph.byId.get('tools')
          const text = additions.length ? additions.join(', ') : removals.length ? '− ' + removals.join(', ') : ''
          if (!node) {
            node = addNode(graph, {
              id: 'tools',
              kind: 'tools',
              sector: 'parietal',
              titleKey: 'node.kind.tools',
              text,
              time,
              weight: 1.05,
              permanent: true,
              hemi: 0,
              status: 'done',
            })
          } else if (text) {
            node.text = clip(node.text ? node.text + '\n' + text : text, CLIP_TEXT)
            node.updatedAt = time
          }
          return
        }
        case 'assistant/message': {
          const message = data.message || {}
          const source = message.source || {}
          if (source.provider || source.model) {
            setModel(graph, source.provider, source.model, { countRequest: true })
          }
          if (data.usage) graph.tokens = data.usage
          removeLive(graph)
          const content = message.content || []
          const reasoning = reasoningOfContent(content)
          for (let index = 0; index < reasoning.length; index += 1) {
            const node = addNode(graph, {
              id: 'reason:' + event.seq + ':' + index,
              kind: 'reasoning',
              sector: 'prefrontal',
              title: firstLine(reasoning[index], 44) || 'reasoning',
              text: reasoning[index],
              time,
              weight: 1.05,
              status: 'done',
            })
            graph.lastReasonId = node.id
            graph.lastActive = node.id
          }
          const answer = textOfContent(content)
          if (answer.trim()) {
            addNode(graph, {
              id: 'answer:' + event.seq,
              kind: 'answer',
              sector: 'occipital',
              title: firstLine(answer, 44) || 'answer',
              text: answer,
              time,
              weight: 1.1,
              status: 'done',
            })
          }
          for (const block of blocksOfType(content, 'tool-call')) {
            upsertTool(graph, block.id, block.name, block.arguments || '', time)
          }
          return
        }
        case 'assistant/attempt': {
          const stream = Array.isArray(data.stream) ? data.stream : []
          const parts = []
          for (const record of stream) {
            const chunk = record && record.chunk ? record.chunk : record
            if (chunk && typeof chunk === 'object') {
              if ((chunk.type === 'reasoning-delta' || chunk.type === 'text-delta') && chunk.text) parts.push(chunk.text)
              if (chunk.type === 'block-end' && chunk.block && typeof chunk.block.text === 'string') parts.push(chunk.block.text)
            }
          }
          const text = parts.join('')
          if (!text.trim()) return
          addNode(graph, {
            id: 'attempt:' + event.seq,
            kind: 'attempt',
            sector: 'prefrontal',
            titleKey: 'node.kind.attempt',
            text,
            time,
            weight: 0.8,
            status: 'error',
          })
          return
        }
        case 'tool/call': {
          const node = upsertTool(graph, data.callId, data.name, data.arguments || '', time)
          if (node) {
            graph.lastActive = node.id
            captureToolTextFiles(graph, data.name, data.arguments || '', node, time)
          }
          return
        }
        case 'tool/result': {
          const message = data.message || {}
          const callId = message.toolCallId || message.source && message.source.callId
          const node = finishTool(
            graph,
            callId,
            !!message.isError,
            textOfContent(message.content),
            data.error && (data.error.reason || data.error.code),
            time,
          )
          captureImages(graph, message.content, node, time, 'tool')
          return
        }
        case 'deliverables/presented': {
          const node = data.callId
            ? graph.byId.get(graph.callIds.get(data.callId) || '')
            : null
          captureLocalFiles(graph, data.files, node, time, 'deliverable')
          captureTextFiles(graph, data.files, node, time, 'deliverable')
          return
        }
        case 'request/context': {
          setModel(graph, data.provider, data.model, { contextWindow: data.contextWindow })
          return
        }
        case 'request/header': {
          const header = data.header || {}
          const config = header.config || {}
          setModel(graph, config.provider, config.model, {
            reasoningEffort: config.reasoningEffort,
            toolCount: Array.isArray(header.tools) ? header.tools.length : undefined,
            countRequest: true,
          })
          return
        }
        default:
          foldUnknown(graph, event, time)
      }
    }

    function foldUnknown(graph, event, time) {
      const type = String(event.type || '')
      if (!type || type.startsWith('plugin:')) return
      let kind = null
      let sector = 'parietal'
      if (/error|failed|abort/i.test(type)) { kind = 'error'; sector = 'stem' }
      else if (/plan|todo/i.test(type)) { kind = 'plan'; sector = 'prefrontal' }
      else if (/goal/i.test(type)) { kind = 'goal'; sector = 'prefrontal' }
      else if (/skill/i.test(type)) { kind = 'reference'; sector = 'temporal' }
      else if (/subagent|agent-team/i.test(type)) { kind = 'agent'; sector = 'cerebellum' }
      else if (/compaction/i.test(type)) { kind = 'document'; sector = 'temporal' }
      else if (/web|fetch|search/i.test(type)) { kind = 'reference'; sector = 'occipital' }
      if (!kind) return
      const text = pickText(event.data)
      addNode(graph, {
        id: 'ev:' + event.seq,
        kind,
        sector,
        title: prettyType(type),
        text,
        time,
        weight: 0.75,
        status: /error|failed/i.test(type) ? 'error' : 'done',
      })
    }

    function prettyType(type) {
      const cleaned = String(type).replace(/^_+/, '').replace(/[\\/]+/g, ' ')
      return clip(cleaned, 42)
    }

    /* ------------------------------------------------------------------ *
     *  Spatial layout of the graph inside the brain
     * ------------------------------------------------------------------ */

    function layoutProjectGraph(graph) {
      const ringNodes = []
      for (const node of graph.nodes) {
        if (node.layout === 'hub' || node.id === 'project') {
          node.p = { x: 0, y: -0.02, z: 0.1 }
          continue
        }
        ringNodes.push(node)
      }
      ringNodes.sort((a, b) => {
        const left = a.ringIndex !== null && a.ringIndex !== undefined ? a.ringIndex : hash(a.id) % 1000
        const right = b.ringIndex !== null && b.ringIndex !== undefined ? b.ringIndex : hash(b.id) % 1000
        return left - right
      })
      ringNodes.forEach((node, index) => {
        const ring = index % 3
        const radius = 0.46 + ring * 0.2
        const angle = index * 2.399963229728653 + ring * 0.75
        let x = Math.cos(angle) * radius
        let z = Math.sin(angle) * radius * 0.82
        let y = ((index % 5) - 2) * 0.065 + (ring - 1) * 0.11
        const norm = Math.sqrt(
          (x / 0.96) * (x / 0.96) + (y / 0.68) * (y / 0.68) + (z / 0.96) * (z / 0.96),
        )
        if (norm > 1) {
          const scale = 0.98 / norm
          x *= scale
          y *= scale
          z *= scale
        }
        node.p = { x, y, z }
      })
    }

    function layoutGraph(graph) {
      if (graph.projectMode) {
        layoutProjectGraph(graph)
        return
      }
      const counters = new Map()
      let promptIndex = 0
      for (const node of graph.nodes) {
        if (node.id === 'root-title') { node.p = { x: 0.02, y: 0.24, z: 0.54 }; continue }
        if (node.id === 'model') { node.p = { x: -0.15, y: -0.55, z: 0.03 }; continue }
        if (node.id === 'system') { node.p = { x: 0.15, y: -0.55, z: 0.03 }; continue }
        if (node.id === 'tools') { node.p = { x: 0.0, y: 0.46, z: -0.08 }; continue }
        if (node.class === 'prompt') {
          /* Prompts sit down the brainstem, below the model/system row. */
          const index = promptIndex
          promptIndex += 1
          const ring = Math.floor(index / 6)
          const angle = index * 2.399963229728653
          const radius = 0.052 + ring * 0.034 + (index % 3) * 0.012
          const x = Math.cos(angle) * radius
          const z = -0.02 + Math.sin(angle) * radius * 0.72
          const y = -0.7 + ((index % 3) - 1) * 0.032
          const nx = x / 0.96
          const ny = y / 0.68
          const nz = z / 0.96
          const norm = Math.sqrt(nx * nx + ny * ny + nz * nz)
          const scale = norm > 0.98 ? 0.98 / norm : 1
          node.p = { x: x * scale, y: y * scale, z: z * scale }
          continue
        }
        const sectorKey = ANCHORS[node.sector] ? node.sector : 'parietal'
        const anchor = ANCHORS[sectorKey]
        const hemi = node.hemi || 0
        const key = sectorKey + ':' + hemi
        const index = counters.get(key) || 0
        counters.set(key, index + 1)
        const angle = index * 2.399963229728653 + (SECTOR_PHASE[sectorKey] || 0)
        const radius = 0.09 + 0.052 * Math.sqrt(index + 1)
        const ox = Math.cos(angle) * radius
        const oy = Math.sin(angle * 1.37) * radius * 0.66
        const oz = Math.sin(angle) * radius * 0.9
        let x = anchor.x * (hemi || 1) + ox
        let y = anchor.y + oy
        let z = anchor.z + oz
        const jitter = (hash(node.id) % 1000) / 1000 - 0.5
        x += jitter * 0.022
        y += jitter * 0.016
        z -= jitter * 0.022
        const nx = x / 0.96
        const ny = y / 0.68
        const nz = z / 0.96
        const norm = Math.sqrt(nx * nx + ny * ny + nz * nz)
        if (norm > 1) {
          const scale = 0.98 / norm
          x *= scale
          y *= scale
          z *= scale
        }
        node.p = { x, y, z }
      }
    }

    function distance3(a, b) {
      const dx = a.p.x - b.p.x
      const dy = a.p.y - b.p.y
      const dz = a.p.z - b.p.z
      return Math.sqrt(dx * dx + dy * dy + dz * dz)
    }

    function buildEdges(graph) {
      const edges = []
      const seen = new Set()
      const push = (a, b, kind) => {
        if (!a || !b || a === b) return
        const key = a < b ? a + '|' + b : b + '|' + a
        if (seen.has(key)) return
        seen.add(key)
        edges.push({ a, b, kind })
      }

      let previous = null
      for (const node of graph.nodes) {
        if (node.id === 'model' || node.id === 'system') continue
        if (previous) push(previous, node.id, 'flow')
        previous = node.id
      }

      for (const node of graph.nodes) {
        if (node.parentId && graph.byId.has(node.parentId)) push(node.parentId, node.id, 'parent')
      }

      const bySector = new Map()
      for (const node of graph.nodes) {
        const list = bySector.get(node.sector) || []
        let best = null
        let second = null
        let bestDistance = Infinity
        let secondDistance = Infinity
        for (const other of list) {
          const distance = distance3(other, node)
          if (distance < bestDistance) {
            second = best
            secondDistance = bestDistance
            best = other
            bestDistance = distance
          } else if (distance < secondDistance) {
            second = other
            secondDistance = distance
          }
        }
        if (best && bestDistance < 0.55) push(best.id, node.id, 'cluster')
        if (second && secondDistance < 0.4) push(second.id, node.id, 'cluster')
        list.push(node)
        bySector.set(node.sector, list)
      }

      if (graph.rootId) {
        if (graph.byId.has('model')) push(graph.rootId, 'model', 'anchor')
        if (graph.byId.has('tools')) push(graph.rootId, 'tools', 'anchor')
      }
      if (graph.byId.has('model') && graph.byId.has('system')) push('model', 'system', 'anchor')

      installEdges(graph, edges)
    }

    function installEdges(graph, edges) {
      graph.edges = edges
      const adjacency = new Map()
      const incoming = new Map()
      const outgoing = new Map()
      for (const edge of edges) {
        if (!adjacency.has(edge.a)) adjacency.set(edge.a, [])
        if (!adjacency.has(edge.b)) adjacency.set(edge.b, [])
        adjacency.get(edge.a).push(edge.b)
        adjacency.get(edge.b).push(edge.a)
        if (!outgoing.has(edge.a)) outgoing.set(edge.a, [])
        outgoing.get(edge.a).push(edge.b)
        if (!incoming.has(edge.b)) incoming.set(edge.b, [])
        incoming.get(edge.b).push(edge.a)
      }
      graph.adj = adjacency
      graph.adjIn = incoming
      graph.adjOut = outgoing
    }

    const PROJECT_MAX_CHATS = 90
    const PROJECT_MAX_CHILDREN = 40

    function projectTitleOfPath(path) {
      const base = basename(path)
      return base || String(path || '')
    }

    /**
     * CWD grouping used when a composition exposes no Workspace registry.
     * Workspace-backed project mode remains the primary path.
     */
    function buildProjectLinks(chats) {
      const byId = new Map(chats.map(chat => [chat.id, chat]))
      const links = []
      const seen = new Set()
      for (const chat of chats) {
        if (!chat.parentId || !byId.has(chat.parentId)) continue
        const key = chat.parentId < chat.id ? chat.parentId + '|' + chat.id : chat.id + '|' + chat.parentId
        if (seen.has(key)) continue
        seen.add(key)
        links.push({
          kind: chat.origin === 'subagent' ? 'subagent' : 'fork',
          a: chat.parentId,
          b: chat.id,
        })
      }
      return links
    }

    function buildProjects(list) {
      const byId = list && list.byId ? list.byId : {}
      const ids = list && Array.isArray(list.ids) ? list.ids : Object.keys(byId)
      const groups = new Map()
      for (const id of ids) {
        const row = byId[id]
        if (!row || row.blank) continue
        const key = row.cwd || ''
        let project = groups.get(key)
        if (!project) {
          project = {
            key,
            title: projectTitleOfPath(key),
            path: key,
            chats: [],
            links: [],
            parents: new Map(),
            children: new Map(),
            updatedAt: 0,
            running: false,
          }
          groups.set(key, project)
        }
        project.chats.push(row)
        project.updatedAt = Math.max(project.updatedAt, row.updatedAt || 0)
        project.running = project.running || !!row.running
      }
      const projects = []
      for (const project of groups.values()) {
        const memberIds = new Set(project.chats.map(chat => chat.id))
        for (const chat of project.chats) {
          if (!chat.parentId || !memberIds.has(chat.parentId)) continue
          if (!project.children.has(chat.parentId)) project.children.set(chat.parentId, [])
          project.children.get(chat.parentId).push(chat.id)
          if (!project.parents.has(chat.id)) project.parents.set(chat.id, [])
          project.parents.get(chat.id).push(chat.parentId)
        }
        project.links = buildProjectLinks(project.chats)
        project.chats.sort((a, b) => {
          if (!!a.running !== !!b.running) return a.running ? -1 : 1
          return (b.updatedAt || 0) - (a.updatedAt || 0)
        })
        projects.push(project)
      }
      return projects.sort((a, b) => {
        if (a.running !== b.running) return a.running ? -1 : 1
        return (b.updatedAt || 0) - (a.updatedAt || 0)
      })
    }

    /**
     * Project mode: the hub is the workspace, each chat is a node, and the
     * links are real lineage (fork / subagent) plus a light activity order.
     * It reuses the brain volume and renderer, not the session lobes.
     */
    function buildProjectGraph(workspace, list) {
      const graph = createGraph()
      graph.projectMode = true
      const byId = list && list.byId ? list.byId : {}
      const workspaceIds = (Array.isArray(workspace.sessionIds) ? workspace.sessionIds : [])
        .filter(id => byId[id] && !byId[id].blank)
      const sorted = workspaceIds.slice().sort((a, b) => {
        const left = byId[a]
        const right = byId[b]
        if (!!left.running !== !!right.running) return left.running ? -1 : 1
        return (right.updatedAt || 0) - (left.updatedAt || 0)
      })
      const includedWorkspace = workspaceIds.length <= PROJECT_MAX_CHATS
        ? workspaceIds
        : sorted.slice(0, PROJECT_MAX_CHATS)
      const includedSet = new Set(includedWorkspace)
      const linked = []
      for (const id of Object.keys(byId)) {
        if (includedSet.has(id)) continue
        const row = byId[id]
        if (row && row.parentId && includedSet.has(row.parentId)) linked.push(id)
      }
      const linkedLimited = linked.slice(0, PROJECT_MAX_CHILDREN)
      const linkedSet = new Set(linkedLimited)
      for (const id of linkedLimited) includedSet.add(id)

      const workspaceSet = new Set(workspaceIds)
      const slotOrder = includedWorkspace.concat(linkedLimited)
      const slotIndex = new Map(slotOrder.map((id, index) => [id, index]))
      const orderedIds = Array.from(includedSet).sort((a, b) => {
        const left = byId[a]
        const right = byId[b]
        if (!!left.running !== !!right.running) return left.running ? -1 : 1
        return (right.updatedAt || 0) - (left.updatedAt || 0)
      })

      let runningCount = 0
      for (const id of includedSet) {
        if (byId[id] && byId[id].running) runningCount += 1
      }

      const hub = addNode(graph, {
        id: 'project',
        kind: 'project',
        sector: 'stem',
        title: workspace.title,
        text: workspace.path,
        summary: workspace.path,
        time: Date.now(),
        weight: 2.0,
        permanent: true,
        hemi: 0,
        layout: 'hub',
        color: '#ffd479',
        status: runningCount ? 'running' : 'done',
      })
      graph.rootId = hub.id

      for (const id of orderedIds) {
        const row = byId[id]
        if (!row) continue
        const child = linkedSet.has(id)
        const node = addNode(graph, {
          id: 'session:' + id,
          kind: child ? 'subagent-session' : 'session',
          sector: 'parietal',
          title: row.displayTitle || String(id).slice(0, 8),
          summary: row.cwd || '',
          text: '',
          time: row.updatedAt || Date.now(),
          updatedAt: row.updatedAt || Date.now(),
          weight: row.running ? 1.5 : 1.05,
          hemi: 0,
          layout: 'ring',
          ringIndex: slotIndex.has(id) ? slotIndex.get(id) : (hash(id) % 1000),
          color: row.running ? '#5dffc0' : child ? '#b78bff' : '#7fb8ff',
          status: row.running ? 'running' : 'done',
          sessionId: id,
          running: !!row.running,
          cwd: row.cwd || '',
          origin: row.origin || null,
          parentId: row.parentId && includedSet.has(row.parentId) ? 'session:' + row.parentId : null,
          meta: row.cwd ? [{ key: 'node.path', value: row.cwd }] : [],
        })
        node.parentSessionId = row.parentId || null
        node.inProject = workspaceSet.has(id)
        node.linkedChild = child
      }

      const edges = []
      const seen = new Set()
      const push = (a, b, kind) => {
        if (!a || !b || a === b) return
        const key = a < b ? a + '|' + b : b + '|' + a
        if (seen.has(key)) return
        seen.add(key)
        edges.push({ a, b, kind })
      }
      for (const id of includedWorkspace) push('project', 'session:' + id, 'anchor')
      for (const id of linkedLimited) {
        const row = byId[id]
        if (row && row.parentId && includedSet.has(row.parentId)) {
          push('session:' + row.parentId, 'session:' + id, 'parent')
        }
      }
      const timeline = includedWorkspace
        .filter(id => byId[id])
        .slice()
        .sort((a, b) => (byId[a].updatedAt || 0) - (byId[b].updatedAt || 0))
      if (timeline.length <= 40) {
        for (let index = 1; index < timeline.length; index += 1) {
          push('session:' + timeline[index - 1], 'session:' + timeline[index], 'time')
        }
      }

      layoutGraph(graph)
      installEdges(graph, edges)
      graph.structureDirty = false
      graph.hasMore = workspaceIds.length > includedWorkspace.length
      graph.windowCount = includedSet.size
      graph.projectMeta = {
        workspaceId: workspace.workspaceId,
        title: workspace.title,
        path: workspace.path,
        total: workspaceIds.length,
        shown: includedWorkspace.length,
        running: runningCount,
        linked: linkedLimited.length,
        connections: edges.length,
        chats: workspaceIds
          .concat(linkedLimited)
          .slice(0, 200)
          .map(id => {
            const row = byId[id]
            if (!row) return null
            return {
              id,
              title: row.displayTitle || String(id).slice(0, 8),
              running: !!row.running,
              updatedAt: row.updatedAt || 0,
              cwd: row.cwd || '',
              origin: row.origin || null,
              parentId: row.parentId || null,
              linked: linkedSet.has(id),
              shown: includedSet.has(id),
            }
          })
          .filter(Boolean),
      }
      return graph
    }

    /* ------------------------------------------------------------------ *
     *  Runtime: retain one Session, fold its journal, publish snapshots
     * ------------------------------------------------------------------ */

    function createRuntime(ctx) {
      const listeners = new Set()
      let graph = createGraph()
      let revision = 0
      let mode = 'chat'
      let projectMeta = null
      let projectTitle = ''
      let sessionId = null
      let sessionTitle = ''
      let running = false
      let loading = false
      let error = null
      let selectedId = null
      let selectionFallback = null
      let reference = null
      let activeBinding = null
      let loadingOlder = false
      let eventsOff = null
      let sessionOff = null
      let generation = 0
      let windowRevision = -1
      let publishTimer = 0
      let lastPublished = 0
      let loadedAt = 0
      let disposed = false
      let playbackSpeedPref = 1
      let playbackLoopPref = false
      const playback = {
        active: false,
        playing: false,
        index: 0,
        total: 0,
        speed: 1,
        loop: false,
        startTime: 0,
        endTime: 0,
        duration: 0,
        graphIndex: 0,
        anchorIndex: 0,
        anchorClock: 0,
        timer: 0,
        entries: [],
        markers: [],
        resumeSelected: null,
      }

      function titleFromList(id) {
        try {
          const list = ctx.sessions.list.getSnapshot()
          const row = list && list.byId ? list.byId[id] : null
          return row ? row.displayTitle : ''
        } catch { return '' }
      }

      function cwdFromList(id) {
        try {
          const list = ctx.sessions.list.getSnapshot()
          const row = list && list.byId ? list.byId[id] : null
          return row && typeof row.cwd === 'string' ? row.cwd : ''
        } catch { return '' }
      }

      function ensureLayout() {
        if (!graph.structureDirty) return
        layoutGraph(graph)
        buildEdges(graph)
        graph.structureDirty = false
      }

      function buildEvents() {
        if (graph.projectMode) return []
        const nodes = graph.nodes.filter(node => EVENT_KINDS.has(node.kind))
        nodes.sort((a, b) => a.time - b.time)
        return nodes.slice(-80).reverse().map(node => ({
          id: node.id,
          kind: node.kind,
          sector: node.sector,
          color: node.color || (node.class === 'prompt' ? CLASSES.prompt.color : null),
          toolName: node.toolName || null,
          family: node.family || null,
          skillName: node.skillName || null,
          mcpServer: node.mcpServer || null,
          mcpTool: node.mcpTool || null,
          title: node.title || '',
          titleKey: node.titleKey || null,
          status: node.status,
          time: node.time,
          live: !!node.live,
          summary: clip(
            node.text
            || node.summary
            || node.result
            || (node.argsRaw ? prettyJson(node.argsRaw) : '')
            || node.error
            || '',
            280,
          ),
        }))
      }

      function buildLive() {
        if (graph.projectMode) return null
        const id = graph.liveReasonId || graph.liveAnswerId
        if (!id) return null
        const node = graph.byId.get(id)
        if (!node) return null
        return {
          id,
          kind: node.kind,
          sector: node.sector,
          color: node.color || null,
          titleKey: node.titleKey || null,
          text: clip(node.text || '', 1800),
          time: node.time,
        }
      }

      function buildFlow() {
        const ids = graph.flowIds.filter(id => graph.byId.has(id))
        const slice = ids.slice(-120)
        const flow = []
        for (let index = 0; index < slice.length; index += 1) {
          const node = graph.byId.get(slice[index])
          const recent = index >= slice.length - 8
          flow.push({
            id: node.id,
            kind: node.kind,
            sector: node.sector,
            title: node.title,
            titleKey: node.titleKey,
            text: clip(node.text, recent ? 4200 : 600),
            time: node.time,
            live: !!node.live,
            status: node.status,
          })
        }
        return flow
      }

      function anchorColor(node) {
        if (!node) return null
        return node.color
          || (CLASSES[node.class] && CLASSES[node.class].color)
          || (SECTORS[node.sector] && SECTORS[node.sector].color)
          || null
      }

      function buildAnchors() {
        const anchors = []
        const root = graph.rootId ? graph.byId.get(graph.rootId) : null
        if (root) anchors.push({ key: 'task', nodeId: root.id, sector: root.sector, color: anchorColor(root), count: 1 })
        const prompts = graph.nodes.filter(node => node.class === 'prompt' && !node.synthetic)
        if (prompts.length) {
          const latest = prompts[prompts.length - 1]
          anchors.push({ key: 'prompt', nodeId: latest.id, sector: latest.sector, color: anchorColor(latest), count: prompts.length })
        }
        const model = graph.byId.get('model')
        if (model) anchors.push({ key: 'model', nodeId: model.id, sector: model.sector, color: anchorColor(model), count: 1 })
        const system = graph.byId.get('system')
        if (system) anchors.push({ key: 'system', nodeId: system.id, sector: system.sector, color: anchorColor(system), count: 1 })
        const docs = graph.nodes.filter(node => node.kind === 'document' || node.kind === 'reference')
        if (docs.length) {
          const latest = docs[docs.length - 1]
          anchors.push({ key: 'docs', nodeId: latest.id, sector: 'temporal', color: anchorColor(latest), count: docs.length })
        }
        const plans = graph.nodes.filter(node => node.kind === 'plan' || node.kind === 'goal')
        if (plans.length) {
          const latest = plans[plans.length - 1]
          anchors.push({ key: 'plan', nodeId: latest.id, sector: 'prefrontal', color: anchorColor(latest), count: plans.length })
        }
        return anchors
      }

      function buildSignals() {
        const tools = graph.nodes.filter(node => node.toolName)
        tools.sort((a, b) => a.time - b.time)
        return tools.slice(-80).reverse().map(node => ({
          id: node.id,
          toolName: node.toolName,
          family: node.family || null,
          skillName: node.skillName || null,
          mcpServer: node.mcpServer || null,
          mcpTool: node.mcpTool || null,
          kind: node.kind,
          sector: node.sector,
          status: node.status,
          summary: clip(node.summary || '', 220),
          time: node.time,
          live: !!node.live,
        }))
      }

      function buildCalls() {
        if (graph.projectMode) return []
        const calls = graph.nodes.filter(node => node.family === 'skill' || node.family === 'mcp')
        calls.sort((a, b) => a.time - b.time)
        return calls.slice(-120).reverse().map(node => ({
          id: node.id,
          family: node.family,
          toolName: node.toolName || '',
          skillName: node.skillName || null,
          mcpServer: node.mcpServer || null,
          mcpTool: node.mcpTool || null,
          title: node.title || '',
          summary: clip(node.summary || (node.argsRaw ? prettyJson(node.argsRaw) : ''), 220),
          status: node.status,
          time: node.time,
          live: !!node.live,
        }))
      }

      function buildCallCounts() {
        const counts = { skill: 0, mcp: 0, total: 0 }
        for (const node of graph.nodes) {
          if (node.family === 'skill') counts.skill += 1
          else if (node.family === 'mcp') counts.mcp += 1
        }
        counts.total = counts.skill + counts.mcp
        return counts
      }

      function describeNode(id) {
        const node = graph.byId.get(id)
        if (!node) return null
        const linkOf = other => ({
          id: other.id,
          kind: other.kind,
          sector: other.sector,
          color: other.color || null,
          title: other.title,
          titleKey: other.titleKey,
          toolName: other.toolName,
          status: other.status,
          time: other.time,
        })
        const collect = source => (source.get(id) || [])
          .map(otherId => graph.byId.get(otherId))
          .filter(Boolean)
          .sort((a, b) => Math.abs(a.time - node.time) - Math.abs(b.time - node.time))
          .slice(0, 10)
          .map(linkOf)
        return {
          id: node.id,
          kind: node.kind,
          sector: node.sector,
          color: node.color || null,
          title: node.title,
          titleKey: node.titleKey,
          text: node.text,
          argsRaw: node.argsRaw,
          result: node.result,
          error: node.error,
          summary: node.summary,
          status: node.status,
          time: node.time,
          updatedAt: node.updatedAt,
          toolName: node.toolName,
          family: node.family || null,
          skillName: node.skillName || null,
          mcpServer: node.mcpServer || null,
          mcpTool: node.mcpTool || null,
          meta: node.meta,
          inputs: collect(graph.adjIn),
          outputs: collect(graph.adjOut),
          media: graph.media
            .filter(item => item.nodeId === node.id)
            .slice(-12)
            .reverse(),
          files: graph.files
            .filter(item => item.nodeId === node.id)
            .slice(-12)
            .reverse(),
          sessionId: node.sessionId || null,
          nodeRunning: !!node.running,
          origin: node.origin || null,
          cwd: node.cwd || '',
          parentId: node.parentId || null,
          parentSessionId: node.parentSessionId || null,
          inProject: !!node.inProject,
          linkedChild: !!node.linkedChild,
          live: !!node.live,
          synthetic: !!node.synthetic,
        }
      }

      function playbackMeta() {
        if (!playback.active) return null
        const total = playback.total
        const index = Math.max(0, Math.min(total, playback.index))
        const at = index > 0 ? playback.entries[index - 1] : playback.entries[0]
        const currentTime = at && at.event ? at.event.time : playback.startTime
        return {
          active: true,
          playing: !!playback.playing,
          index,
          total,
          speed: playback.speed,
          loop: !!playback.loop,
          progress: total ? index / total : 0,
          currentTime,
          startTime: playback.startTime,
          endTime: playback.endTime,
          duration: playback.duration,
          atEnd: index >= total,
          markers: playback.markers,
        }
      }

      function playbackMarkerKind(entry) {
        const event = entry && entry.event
        if (!event) return null
        if (event.type === 'user/message') {
          const source = event.data && event.data.source
          return source && source.kind === 'user' ? 'prompt' : null
        }
        if (event.type === 'assistant/message') {
          const content = event.data && event.data.message && event.data.message.content
          if (Array.isArray(content)) {
            if (content.some(block => block && block.type === 'reasoning')) return 'thought'
            if (content.some(block => block && block.type === 'tool-call')) return 'tool'
            if (content.some(block => block && block.type === 'text')) return 'answer'
          }
          return 'thought'
        }
        if (event.type === 'assistant/attempt') return 'thought'
        if (event.type === 'tool/result') return 'tool'
        if (event.type === 'deliverables/presented') return 'document'
        return null
      }

      function buildPlaybackMarkers(entries) {
        const markers = []
        for (let index = 0; index < entries.length; index += 1) {
          const kind = playbackMarkerKind(entries[index])
          if (!kind) continue
          markers.push({ index: index + 1, time: entries[index].event.time, kind })
        }
        if (markers.length > 120) {
          const stride = Math.ceil(markers.length / 120)
          const sampled = markers.filter((marker, index) => index % stride === 0)
          if (sampled[sampled.length - 1] !== markers[markers.length - 1]) sampled.push(markers[markers.length - 1])
          return sampled.slice(0, 120)
        }
        return markers
      }

      function stopPlaybackTimer() {
        if (playback.timer) {
          clearInterval(playback.timer)
          playback.timer = 0
        }
      }

      function clearPlaybackState() {
        stopPlaybackTimer()
        playback.active = false
        playback.playing = false
        playback.index = 0
        playback.total = 0
        playback.startTime = 0
        playback.endTime = 0
        playback.duration = 0
        playback.graphIndex = 0
        playback.anchorIndex = 0
        playback.anchorClock = 0
        playback.entries = []
        playback.markers = []
        playback.resumeSelected = null
      }

      function setPlaybackGraphIndex(target) {
        if (!playback.active) return
        const next = Math.max(0, Math.min(playback.total, Math.round(target)))
        if (next === playback.graphIndex) {
          playback.index = next
          return
        }
        if (next < playback.graphIndex) {
          resetGraph(graph)
          graph.windowCount = 0
          for (let index = 0; index < next; index += 1) foldEntry(graph, playback.entries[index])
        } else {
          for (let index = playback.graphIndex; index < next; index += 1) foldEntry(graph, playback.entries[index])
        }
        playback.graphIndex = next
        playback.index = next
        graph.windowCount = next
        graph.hasMore = false
        graph.loadedAt = Date.now()
        if (selectedId && !graph.byId.has(selectedId)) selectedId = null
        schedule()
      }

      function playbackAdvance() {
        if (!playback.active || !playback.playing) return
        const total = Math.max(1, playback.total)
        const perEntry = playback.duration / total / Math.max(0.25, playback.speed || 1)
        const target = playback.anchorIndex + Math.floor((Date.now() - playback.anchorClock) / perEntry)
        const next = Math.min(total, target)
        if (next > playback.index) setPlaybackGraphIndex(next)
        if (next >= total) {
          if (playback.loop) {
            playback.index = 0
            setPlaybackGraphIndex(0)
            playback.anchorIndex = 0
            playback.anchorClock = Date.now()
          } else {
            playback.playing = false
            stopPlaybackTimer()
            emit()
          }
        }
      }

      function startPlaybackTimer() {
        stopPlaybackTimer()
        if (!playback.active || !playback.playing) return
        playback.anchorIndex = playback.index
        playback.anchorClock = Date.now()
        playback.timer = setInterval(playbackAdvance, 70)
      }

      function startPlayback() {
        if (disposed || mode === 'project' || !activeBinding || playback.active) return
        const window = activeBinding.eventSource.getSnapshot()
        const entries = (window && Array.isArray(window.entries) ? window.entries : [])
          .filter(entry => entry && entry.type === 'event')
        if (!entries.length) return
        playback.entries = entries
        playback.total = entries.length
        playback.index = 0
        playback.graphIndex = 0
        playback.speed = playbackSpeedPref
        playback.loop = playbackLoopPref
        playback.startTime = entries[0].event.time || 0
        playback.endTime = entries[entries.length - 1].event.time || playback.startTime
        playback.duration = Math.max(4000, Math.min(45000, entries.length * 90))
        playback.markers = buildPlaybackMarkers(entries)
        playback.active = true
        playback.playing = true
        playback.resumeSelected = selectedId
        selectedId = null
        resetGraph(graph)
        graph.windowCount = 0
        graph.hasMore = false
        graph.loadedAt = Date.now()
        loading = false
        error = null
        startPlaybackTimer()
        emit()
      }

      function resumePlayback() {
        if (!playback.active) {
          startPlayback()
          return
        }
        if (playback.playing) return
        playback.playing = true
        if (playback.index >= playback.total) {
          playback.index = 0
          setPlaybackGraphIndex(0)
        }
        startPlaybackTimer()
        emit()
      }

      function pausePlayback() {
        if (!playback.active || !playback.playing) return
        playbackAdvance()
        playback.playing = false
        stopPlaybackTimer()
        emit()
      }

      function togglePlayback() {
        if (!playback.active || !playback.playing) resumePlayback()
        else pausePlayback()
      }

      function seekPlayback(index) {
        if (!playback.active) return
        const next = Math.max(0, Math.min(playback.total, Number(index) || 0))
        setPlaybackGraphIndex(next)
        if (playback.playing) {
          if (next >= playback.total) {
            playback.playing = false
            stopPlaybackTimer()
          } else {
            playback.anchorIndex = next
            playback.anchorClock = Date.now()
          }
        }
        emit()
      }

      function stepPlayback(delta) {
        if (!playback.active) return
        pausePlayback()
        const next = Math.max(0, Math.min(playback.total, playback.index + Math.sign(delta || 1)))
        setPlaybackGraphIndex(next)
        playback.anchorIndex = next
        playback.anchorClock = Date.now()
        emit()
      }

      function restartPlayback() {
        if (!playback.active) {
          startPlayback()
          return
        }
        playback.playing = true
        playback.index = 0
        setPlaybackGraphIndex(0)
        startPlaybackTimer()
        emit()
      }

      function setPlaybackSpeed(value) {
        const speed = Number(value)
        if (![0.5, 1, 2, 4].includes(speed)) return
        playbackSpeedPref = speed
        if (playback.active) {
          playback.speed = speed
          if (playback.playing) {
            playback.anchorIndex = playback.index
            playback.anchorClock = Date.now()
          }
        }
        emit()
      }

      function togglePlaybackLoop() {
        playbackLoopPref = !playbackLoopPref
        if (playback.active) playback.loop = playbackLoopPref
        emit()
      }

      function exitPlayback() {
        if (!playback.active) return
        stopPlaybackTimer()
        playback.active = false
        playback.playing = false
        const resume = playback.resumeSelected
        if (activeBinding) {
          try {
            const window = activeBinding.eventSource.getSnapshot()
            foldWindow(graph, window)
            windowRevision = window.revision
            sessionTitle = titleFromList(sessionId)
            ensureTaskAnchor(graph, sessionTitle, sessionId)
            selectedId = resume && graph.byId.has(resume) ? resume : null
            loading = false
            error = null
            try { running = !!activeBinding.session.getSnapshot().running } catch { running = false }
          } catch (failure) {
            error = errorText(failure)
          }
        }
        clearPlaybackState()
        emit()
      }

      function build() {
        ensureLayout()
        if (selectedId && !graph.byId.has(selectedId)) {
          if (selectionFallback) {
            const candidates = graph.nodes.filter(node => node.kind === selectionFallback)
            selectedId = candidates.length ? candidates[candidates.length - 1].id : null
            selectionFallback = null
          } else {
            selectedId = null
          }
        }
        const counts = {}
        for (const key of SECTOR_KEYS) counts[key] = 0
        const classCounts = {}
        for (const key of CLASS_KEYS) classCounts[key] = 0
        for (const node of graph.nodes) {
          if (counts[node.sector] === undefined) counts[node.sector] = 0
          counts[node.sector] += 1
          if (classCounts[node.class] === undefined) classCounts[node.class] = 0
          classCounts[node.class] += 1
        }
        revision += 1
        const projectMode = mode === 'project'
        const playbackActive = playback.active
        const project = projectMode ? projectMeta : null
        return {
          revision,
          mode,
          sessionId: projectMode ? null : sessionId,
          title: projectMode ? projectTitle : (sessionTitle || (sessionId ? String(sessionId).slice(0, 8) : '')),
          running: projectMode ? !!(project && project.running) : (playbackActive ? false : running),
          loading,
          error,
          loadedAt,
          loadingOlder,
          hasMore: projectMode ? !!(project && project.total > project.shown) : (playbackActive ? false : graph.hasMore),
          eventCount: projectMode && project ? project.total : graph.windowCount,
          nodeCount: graph.nodes.length,
          edgeCount: graph.edges.length,
          capped: !projectMode && graph.nodes.length >= MAX_NODES,
          dropped: graph.dropped || 0,
          sectors: SECTOR_KEYS.map(key => ({ key, count: counts[key] || 0 })),
          classes: CLASS_KEYS.map(key => ({ key, count: classCounts[key] || 0, color: CLASSES[key].color })),
          model: projectMode ? null : graph.modelInfo,
          tokens: graph.tokens,
          tokenTotal: projectMode ? 0 : tokensOf(graph.tokens),
          toolCount: projectMode && project ? project.connections : graph.toolCount,
          flow: projectMode ? [] : buildFlow(),
          anchors: projectMode ? [] : buildAnchors(),
          signals: projectMode ? [] : buildSignals(),
          calls: projectMode ? [] : buildCalls(),
          callCounts: projectMode ? { skill: 0, mcp: 0, total: 0 } : buildCallCounts(),
          events: playbackActive ? [] : buildEvents(),
          live: playbackActive ? null : buildLive(),
          media: projectMode
            ? []
            : graph.media.slice().sort((a, b) => b.time - a.time),
          mediaEvents: projectMode || playbackActive
            ? []
            : graph.mediaEvents.slice().sort((a, b) => b.time - a.time),
          mediaDropped: graph.mediaDropped || 0,
          files: projectMode
            ? []
            : graph.files.slice().sort((a, b) => b.time - a.time),
          fileEvents: projectMode || playbackActive
            ? []
            : graph.fileEvents.slice().sort((a, b) => b.time - a.time),
          filesDropped: graph.filesDropped || 0,
          playback: playbackMeta(),
          playbackSpeed: playbackSpeedPref,
          playbackLoop: playbackLoopPref,
          project,
          selectedId,
          selected: selectedId ? describeNode(selectedId) : null,
          lastActive: graph.lastActive,
          lastChunkAt: graph.lastChunkAt,
        }
      }

      function emit() {
        if (publishTimer) {
          clearTimeout(publishTimer)
          publishTimer = 0
        }
        lastPublished = Date.now()
        snapshot = build()
        for (const listener of [...listeners]) {
          try { listener() } catch (failure) { console.error(failure) }
        }
      }

      let snapshot = build()

      function schedule() {
        if (publishTimer) return
        const wait = Math.max(70, 120 - (Date.now() - lastPublished))
        publishTimer = setTimeout(() => {
          publishTimer = 0
          emit()
        }, wait)
      }

      function releaseSession() {
        clearPlaybackState()
        generation += 1
        if (eventsOff) { try { eventsOff() } catch {} eventsOff = null }
        if (sessionOff) { try { sessionOff() } catch {} sessionOff = null }
        if (reference) { try { reference.release() } catch {} reference = null }
        activeBinding = null
        loadingOlder = false
        windowRevision = -1
      }

      function attach(id) {
        if (disposed) return
        if (mode === 'chat' && reference && id === sessionId) return
        releaseSession()
        mode = 'chat'
        projectMeta = null
        projectTitle = ''
        sessionId = id
        graph = createGraph()
        graph.cwd = cwdFromList(id)
        selectedId = null
        selectionFallback = null
        loading = true
        error = null
        loadedAt = Date.now()
        sessionTitle = titleFromList(id)
        running = false
        windowRevision = -1
        emit()

        const myGeneration = ++generation
        let ref
        try {
          const address = typeof ctx.sessions.subagentAddress === 'function'
            ? ctx.sessions.subagentAddress(id)
            : undefined
          const target = address || id
          ref = ctx.sessions.retain(target, { source: 'neural-thoughts' })
        } catch (failure) {
          loading = false
          error = errorText(failure)
          emit()
          return
        }
        reference = ref
        ref.ready.then(binding => {
          if (disposed || myGeneration !== generation) return
          activeBinding = binding
          loading = false
          const source = binding.eventSource
          const applyWindow = () => {
            if (disposed || myGeneration !== generation) return
            if (playback.active) return
            const window = source.getSnapshot()
            if (window.revision === windowRevision) return
            if (windowRevision >= 0 && window.revision === windowRevision + 1 && window.change && window.change.kind === 'append') {
              for (const entry of window.change.entries) foldEntry(graph, entry)
              graph.windowCount = window.entries.length
              graph.hasMore = !!window.hasMore
            } else {
              foldWindow(graph, window)
            }
            windowRevision = window.revision
            sessionTitle = titleFromList(id)
            ensureTaskAnchor(graph, sessionTitle, id)
            try {
              const sessionSnapshot = binding.session.getSnapshot()
              running = !!sessionSnapshot.running
            } catch { running = false }
            schedule()
          }
          eventsOff = source.subscribe(applyWindow)
          sessionOff = binding.session.subscribe(() => {
            if (disposed || myGeneration !== generation) return
            try {
              const next = !!binding.session.getSnapshot().running
              if (next !== running) { running = next; schedule() }
            } catch {}
          })
          sessionTitle = titleFromList(id)
          try { running = !!binding.session.getSnapshot().running } catch {}
          applyWindow()
          emit()
        }).catch(failure => {
          if (disposed || myGeneration !== generation) return
          loading = false
          error = errorText(failure)
          emit()
        })
      }

      async function loadOlderHistory() {
        if (disposed || playback.active || mode === 'project' || !activeBinding || loadingOlder || !graph.hasMore) return
        const session = activeBinding.session
        if (!session || typeof session.loadOlder !== 'function') return
        loadingOlder = true
        emit()
        try {
          await session.loadOlder()
        } catch {
          /* No progress is not a fatal panel error; the Session keeps its own status. */
        } finally {
          loadingOlder = false
          schedule()
        }
      }

      function detach(silent) {
        releaseSession()
        mode = 'chat'
        projectMeta = null
        projectTitle = ''
        sessionId = null
        sessionTitle = ''
        running = false
        loading = false
        error = null
        selectedId = null
        selectionFallback = null
        loadedAt = 0
        graph = createGraph()
        if (!silent) emit()
      }

      function showProject(workspace, list) {
        if (disposed || !workspace) return
        if (playback.active) clearPlaybackState()
        const sameWorkspace = mode === 'project'
          && projectMeta
          && projectMeta.workspaceId === workspace.workspaceId
        if (!sameWorkspace) {
          releaseSession()
          selectedId = null
          selectionFallback = null
        }
        const previousSelection = selectedId
        const previousEpoch = graph.epoch || 0
        mode = 'project'
        sessionId = null
        sessionTitle = ''
        running = false
        loading = false
        error = null
        loadedAt = Date.now()
        projectTitle = workspace.title
        try {
          graph = buildProjectGraph(workspace, list)
          graph.epoch = previousEpoch + 1
          projectMeta = graph.projectMeta
          selectedId = previousSelection && graph.byId.has(previousSelection) ? previousSelection : null
        } catch (failure) {
          graph = createGraph()
          graph.projectMode = true
          projectMeta = {
            workspaceId: workspace.workspaceId,
            title: workspace.title,
            path: workspace.path,
            total: 0,
            shown: 0,
            running: 0,
            linked: 0,
            connections: 0,
            chats: [],
          }
          selectedId = null
          error = errorText(failure)
        }
        emit()
      }

      function selectNode(id) {
        const next = id && graph.byId.has(id) ? id : null
        const node = next ? graph.byId.get(next) : null
        selectionFallback = node && node.live && (node.kind === 'reasoning' || node.kind === 'answer')
          ? node.kind
          : null
        if (next === selectedId) return
        selectedId = next
        emit()
      }

      function dispose() {
        disposed = true
        clearPlaybackState()
        if (publishTimer) { clearTimeout(publishTimer); publishTimer = 0 }
        detach(true)
        listeners.clear()
      }

      return {
        getSnapshot() { return snapshot },
        subscribe(listener) {
          listeners.add(listener)
          return () => { listeners.delete(listener) }
        },
        currentGraph() { return graph },
        select(id) {
          if (id == null) return detach()
          attach(id)
        },
        showProject(workspace, list) {
          showProject(workspace, list)
        },
        setActive(active) {
          if (!active) detach()
        },
        selectNode,
        reload() {
          if (sessionId) {
            const id = sessionId
            detach(true)
            attach(id)
          }
        },
        loadOlder: loadOlderHistory,
        playbackStart: startPlayback,
        playbackToggle: togglePlayback,
        playbackPause: pausePlayback,
        playbackResume: resumePlayback,
        playbackSeek: seekPlayback,
        playbackStep: stepPlayback,
        playbackRestart: restartPlayback,
        playbackSpeed: setPlaybackSpeed,
        playbackLoop: togglePlaybackLoop,
        playbackExit: exitPlayback,
        dispose,
      }
    }

    /* ------------------------------------------------------------------ *
     *  Brain point cloud
     * ------------------------------------------------------------------ */

    function mulberry32(seed) {
      let value = seed >>> 0
      return function random() {
        value += 0x6D2B79F5
        let t = value
        t = Math.imul(t ^ (t >>> 15), t | 1)
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296
      }
    }

    function makeBrainPoints() {
      const random = mulberry32(20261115)
      const points = []

      for (const side of [-1, 1]) {
        const count = 1250
        for (let index = 0; index < count; index += 1) {
          const y = 1 - (index / (count - 1)) * 2
          const ring = Math.sqrt(Math.max(0, 1 - y * y))
          const theta = index * 2.399963229728653
          const x = Math.cos(theta) * ring
          const z = Math.sin(theta) * ring
          const ripple = 1
            + 0.048 * Math.sin(7 * theta + 3.1 * y * Math.PI)
            + 0.027 * Math.sin(11.3 * theta * 0.7 + 5.2 * y)
          let px = x * 0.56 * ripple + 0.32 * side
          /* Keep a thin longitudinal fissure so the two hemispheres read apart. */
          if (Math.abs(px) < 0.055) px += 0.06 * side
          const py = y * 0.45 * ripple + 0.02
          const pz = z * 0.76 * ripple * (1 + 0.05 * (0.5 + 0.5 * y))
          const tone = random()
          points.push({
            x: px, y: py, z: pz,
            b: random(),
            tone: tone > 0.93 ? 2 : tone > 0.7 ? 1 : 0,
            phase: random() * TAU,
          })
        }
      }

      for (let index = 0; index < 430; index += 1) {
        const u = random() * 2 - 1
        const angle = random() * TAU
        const radius = Math.cbrt(random())
        const ring = Math.sqrt(Math.max(0, 1 - u * u))
        const x = Math.cos(angle) * ring * radius
        const y = u * radius
        const z = Math.sin(angle) * ring * radius
        const ripple = 1 + 0.1 * Math.sin(9 * Math.atan2(z, x) + 8 * y * Math.PI)
        points.push({
          x: x * 0.24 * ripple,
          y: -0.44 + y * 0.15 * ripple,
          z: -0.56 + z * 0.22 * ripple,
          b: random(),
          tone: 0,
          phase: random() * TAU,
        })
      }

      for (let index = 0; index < 240; index += 1) {
        const t = random()
        const angle = random() * TAU
        const radius = 0.1 * (1 - 0.35 * t)
        points.push({
          x: Math.cos(angle) * radius,
          y: -0.3 - 0.5 * t,
          z: -0.1 + Math.sin(angle) * radius + 0.09 * t * t,
          b: random(),
          tone: 0,
          phase: random() * TAU,
        })
      }

      return points
    }

    /* ------------------------------------------------------------------ *
     *  Holographic shell wireframe
     * ------------------------------------------------------------------ */

    function makeShellLines() {
      const lines = []
      const latSteps = [-58, -29, 0, 29, 58]
      const lonSteps = 6
      for (const side of [-1, 1]) {
        const cx = 0.32 * side
        const rx = 0.56
        const ry = 0.45
        const rz = 0.76
        for (const lat of latSteps) {
          const phi = lat * Math.PI / 180
          const ring = []
          for (let step = 0; step <= 54; step += 1) {
            const theta = (step / 54) * TAU
            let x = cx + rx * Math.cos(phi) * Math.cos(theta)
            const y = 0.02 + ry * Math.sin(phi)
            const z = rz * Math.cos(phi) * Math.sin(theta)
            if (Math.abs(x) < 0.055) x += 0.06 * side
            ring.push({ x, y, z })
          }
          lines.push(ring)
        }
        for (let index = 0; index < lonSteps; index += 1) {
          const theta = (index / lonSteps) * Math.PI
          const arc = []
          for (let step = 0; step <= 40; step += 1) {
            const phi = (step / 40) * Math.PI - Math.PI / 2
            let x = cx + rx * Math.cos(phi) * Math.cos(theta)
            const y = 0.02 + ry * Math.sin(phi)
            const z = rz * Math.cos(phi) * Math.sin(theta)
            if (Math.abs(x) < 0.055) x += 0.06 * side
            arc.push({ x, y, z })
          }
          lines.push(arc)
        }
      }
      for (const lat of [-46, -18, 12]) {
        const phi = lat * Math.PI / 180
        const ring = []
        for (let step = 0; step <= 32; step += 1) {
          const theta = (step / 32) * TAU
          ring.push({
            x: Math.cos(phi) * Math.cos(theta) * 0.24,
            y: -0.44 + Math.sin(phi) * 0.15,
            z: -0.56 + Math.cos(phi) * Math.sin(theta) * 0.22,
          })
        }
        lines.push(ring)
      }
      for (let index = 0; index < 4; index += 1) {
        const theta = (index / 4) * Math.PI
        const arc = []
        for (let step = 0; step <= 18; step += 1) {
          const phi = (step / 18) * Math.PI - Math.PI / 2
          arc.push({
            x: Math.cos(phi) * Math.cos(theta) * 0.24,
            y: -0.44 + Math.sin(phi) * 0.15,
            z: -0.56 + Math.cos(phi) * Math.sin(theta) * 0.22,
          })
        }
        lines.push(arc)
      }
      for (let index = 0; index < 3; index += 1) {
        const angle = (index / 3) * TAU
        const line = []
        for (let step = 0; step <= 16; step += 1) {
          const t = step / 16
          const radius = 0.1 * (1 - 0.35 * t)
          line.push({
            x: Math.cos(angle) * radius,
            y: -0.3 - 0.5 * t,
            z: -0.1 + Math.sin(angle) * radius + 0.09 * t * t,
          })
        }
        lines.push(line)
      }
      return lines
    }

    /* ------------------------------------------------------------------ *
     *  3D renderer
     * ------------------------------------------------------------------ */

    function hexToRgba(hex, alpha) {
      const value = String(hex).replace('#', '')
      const full = value.length === 3 ? value.split('').map(part => part + part).join('') : value
      const number = parseInt(full, 16)
      const r = (number >> 16) & 255
      const g = (number >> 8) & 255
      const b = number & 255
      return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')'
    }

    function makeGlowSprite(color) {
      const size = 128
      const canvas = document.createElement('canvas')
      canvas.width = size
      canvas.height = size
      const context = canvas.getContext('2d')
      const gradient = context.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
      gradient.addColorStop(0, 'rgba(255,255,255,0.95)')
      gradient.addColorStop(0.16, hexToRgba(color, 0.86))
      gradient.addColorStop(0.42, hexToRgba(color, 0.26))
      gradient.addColorStop(0.72, hexToRgba(color, 0.06))
      gradient.addColorStop(1, hexToRgba(color, 0))
      context.fillStyle = gradient
      context.fillRect(0, 0, size, size)
      return canvas
    }

    function createRenderer(canvas, options) {
      const context = canvas.getContext('2d')
      const opts = options || {}
      const points = makeBrainPoints()
      const shellLines = makeShellLines()
      const sprites = new Map()
      const whiteSprite = makeGlowSprite('#bfe8ff')
      for (const key of SECTOR_KEYS) sprites.set(key, makeGlowSprite(SECTORS[key].color))

      const BASE_COLORS = ['#5f9dff', '#9a7bff', '#61e3ff']
      const AMBIENT = []
      const ambientRandom = mulberry32(97531)
      for (let index = 0; index < 90; index += 1) {
        const angle = ambientRandom() * TAU
        const radius = 0.55 + ambientRandom() * 0.85
        const height = (ambientRandom() - 0.5) * 1.2
        AMBIENT.push({
          x: Math.cos(angle) * radius,
          y: height,
          z: Math.sin(angle) * radius,
          phase: ambientRandom() * TAU,
          speed: 0.15 + ambientRandom() * 0.4,
        })
      }

      const RINGS = [
        { radius: 1.02, squash: 0.16, depth: 0.96, tilt: 0.1, speed: 0.05 },
        { radius: 1.12, squash: 0.74, depth: 0.6, tilt: -0.36, speed: -0.04 },
        { radius: 0.94, squash: 0.54, depth: 0.8, tilt: 0.52, speed: 0.03 },
      ]

      let width = 1
      let height = 1
      let dpr = 1
      let yaw = 0.45
      let targetYaw = 0.45
      let pitch = -0.1
      let targetPitch = -0.1
      let zoom = 1
      let targetZoom = 1
      let dragging = false
      let pointerId = null
      let downX = 0
      let downY = 0
      let lastX = 0
      let lastY = 0
      let moved = 0
      let downAt = 0
      let holdUntil = 0
      let hoverId = null
      let selectedId = null
      let focusSector = null
      let classFilter = null
      let showLabels = true
      let graphSource = () => null
      let epoch = -1
      let birthCursor = 0
      let pulses = []
      let hits = []
      let raf = 0
      let destroyed = false
      let lastFrame = 0
      const reducedMotion = typeof window !== 'undefined'
        && typeof window.matchMedia === 'function'
        && window.matchMedia('(prefers-reduced-motion: reduce)').matches

      function resize() {
        const parent = canvas.parentElement
        const rect = parent ? parent.getBoundingClientRect() : canvas.getBoundingClientRect()
        width = Math.max(1, Math.round(rect.width))
        height = Math.max(1, Math.round(rect.height))
        dpr = Math.min(2, window.devicePixelRatio || 1)
        canvas.width = Math.max(1, Math.round(width * dpr))
        canvas.height = Math.max(1, Math.round(height * dpr))
      }

      function rotatePoint(point) {
        const cosYaw = Math.cos(yaw)
        const sinYaw = Math.sin(yaw)
        const x1 = point.x * cosYaw + point.z * sinYaw
        const z1 = -point.x * sinYaw + point.z * cosYaw
        const cosPitch = Math.cos(pitch)
        const sinPitch = Math.sin(pitch)
        return {
          x: x1,
          y: point.y * cosPitch - z1 * sinPitch,
          z: point.y * sinPitch + z1 * cosPitch,
        }
      }

      function project(point) {
        const factor = 3.15 / (3.15 - point.z * 1.05)
        const scale = Math.min(width, height) * 0.46 * zoom
        return {
          x: width / 2 + point.x * scale * factor,
          y: height / 2 - point.y * scale * factor,
          k: factor,
        }
      }

      function blit(sprite, x, y, size, alpha) {
        if (!(alpha > 0) || size <= 0) return
        context.globalAlpha = alpha
        context.drawImage(sprite, x - size / 2, y - size / 2, size, size)
        context.globalAlpha = 1
      }

      function roundRectPath(x, y, w, h, radius) {
        const r = Math.min(radius, h / 2, w / 2)
        context.beginPath()
        context.moveTo(x + r, y)
        context.arcTo(x + w, y, x + w, y + h, r)
        context.arcTo(x + w, y + h, x, y + h, r)
        context.arcTo(x, y + h, x, y, r)
        context.arcTo(x, y, x + w, y, r)
        context.closePath()
      }

      function drawCloud(now) {
        const buckets = []
        for (let index = 0; index < 12; index += 1) buckets.push([])
        for (const point of points) {
          const rotated = rotatePoint(point)
          const projected = project(rotated)
          if (projected.x < -40 || projected.x > width + 40 || projected.y < -40 || projected.y > height + 40) continue
          const depth = clamp((rotated.z + 1.1) / 2.2, 0, 1)
          const shimmer = 0.5 + 0.5 * Math.sin(now * 0.0011 + point.phase)
          const bucket = clamp(Math.floor(depth * 3.2 + shimmer * 0.7), 0, 3)
          const size = point.b > 0.975 ? 2.4 : point.b > 0.86 ? 1.6 : 1.1
          const index = point.tone * 4 + bucket
          buckets[index].push(projected.x, projected.y, size)
        }
        const alphas = [0.08, 0.16, 0.26, 0.4]
        for (let tone = 0; tone < 3; tone += 1) {
          for (let bucket = 0; bucket < 4; bucket += 1) {
            const values = buckets[tone * 4 + bucket]
            if (!values.length) continue
            context.fillStyle = BASE_COLORS[tone]
            context.globalAlpha = alphas[bucket]
            context.beginPath()
            for (let index = 0; index < values.length; index += 3) {
              context.rect(values[index], values[index + 1], values[index + 2], values[index + 2])
            }
            context.fill()
          }
        }
        context.globalAlpha = 1
      }

      function drawShell(now) {
        context.globalCompositeOperation = 'lighter'
        context.lineWidth = 0.7
        for (let index = 0; index < shellLines.length; index += 1) {
          const line = shellLines[index]
          const depthPhase = Math.sin(now * 0.0004 + index * 0.7)
          context.strokeStyle = 'rgba(112, 190, 255, ' + (0.05 + 0.022 * depthPhase).toFixed(3) + ')'
          context.beginPath()
          for (let step = 0; step < line.length; step += 1) {
            const projected = project(rotatePoint(line[step]))
            if (step === 0) context.moveTo(projected.x, projected.y)
            else context.lineTo(projected.x, projected.y)
          }
          context.stroke()
        }
        context.globalCompositeOperation = 'source-over'
      }

      function drawAmbient(now) {
        context.fillStyle = '#8fd8ff'
        for (const particle of AMBIENT) {
          const drifted = {
            x: particle.x,
            y: particle.y + Math.sin(now * 0.0004 * particle.speed + particle.phase) * 0.08,
            z: particle.z,
          }
          const rotated = rotatePoint(drifted)
          const projected = project(rotated)
          if (projected.x < 0 || projected.x > width || projected.y < 0 || projected.y > height) continue
          const alpha = 0.05 + 0.1 * (0.5 + 0.5 * Math.sin(now * 0.0009 + particle.phase))
          context.globalAlpha = alpha
          context.fillRect(projected.x, projected.y, 1.2, 1.2)
        }
        context.globalAlpha = 1
      }

      function drawRings(now) {
        context.globalCompositeOperation = 'lighter'
        context.lineWidth = 1
        for (const ring of RINGS) {
          context.strokeStyle = 'rgba(122, 200, 255, 0.16)'
          context.globalAlpha = 0.35
          context.setLineDash([2, 8])
          context.beginPath()
          for (let step = 0; step <= 96; step += 1) {
            const angle = (step / 96) * TAU + now * 0.00004 * ring.speed * 60
            const point = {
              x: Math.cos(angle) * ring.radius,
              y: Math.sin(angle) * ring.squash + ring.tilt * 0.16,
              z: Math.sin(angle) * ring.depth,
            }
            const rotated = rotatePoint(point)
            const projected = project(rotated)
            if (step === 0) context.moveTo(projected.x, projected.y)
            else context.lineTo(projected.x, projected.y)
          }
          context.stroke()
        }
        context.setLineDash([])
        context.globalAlpha = 1
        context.globalCompositeOperation = 'source-over'
      }

      function nodeColor(node) {
        if (node.family === 'skill' || node.family === 'mcp') return callFamilyColor(node.family)
        return node.color || (SECTORS[node.sector] || SECTORS.parietal).color
      }

      function drawCallGlyph(family, x, y, radius, color, strength, selected) {
        const spread = radius + (selected ? 7 : 5.2)
        context.save()
        context.globalAlpha = clamp(strength, 0, 1)
        context.strokeStyle = color
        context.fillStyle = hexToRgba(color, (selected ? 0.16 : 0.08) * strength)
        context.lineWidth = selected ? 1.8 : 1.25
        context.beginPath()
        if (family === 'skill') {
          for (let index = 0; index < 6; index += 1) {
            const angle = -Math.PI / 2 + index * TAU / 6
            const px = x + Math.cos(angle) * spread
            const py = y + Math.sin(angle) * spread
            if (index === 0) context.moveTo(px, py)
            else context.lineTo(px, py)
          }
        } else {
          roundRectPath(x - spread, y - spread * 0.82, spread * 2, spread * 1.64, spread * 0.34)
        }
        context.closePath()
        context.fill()
        context.stroke()
        if (family === 'skill') {
          context.beginPath()
          context.moveTo(x, y - radius * 0.62)
          context.lineTo(x + radius * 0.54, y + radius * 0.34)
          context.lineTo(x - radius * 0.54, y + radius * 0.34)
          context.closePath()
          context.stroke()
        } else {
          context.setLineDash([2, 3])
          context.beginPath()
          context.moveTo(x - spread * 0.58, y - spread * 0.46)
          context.lineTo(x + spread * 0.58, y + spread * 0.46)
          context.moveTo(x + spread * 0.58, y - spread * 0.46)
          context.lineTo(x - spread * 0.58, y + spread * 0.46)
          context.stroke()
          context.setLineDash([])
        }
        context.restore()
      }

      function drawCore(now) {
        const center = project(rotatePoint({ x: 0, y: 0, z: 0 }))
        context.globalCompositeOperation = 'lighter'
        blit(whiteSprite, center.x, center.y, Math.min(width, height) * 0.95, 0.06)
        context.globalCompositeOperation = 'source-over'
      }

      function drawSectorHalos(now) {
        const graph = graphSource()
        if (!graph || graph.projectMode) return
        const counts = new Map()
        for (const node of graph.nodes) counts.set(node.sector, (counts.get(node.sector) || 0) + 1)
        context.globalCompositeOperation = 'lighter'
        for (const key of SECTOR_KEYS) {
          if (!counts.get(key)) continue
          const anchor = ANCHORS[key]
          const sprite = sprites.get(key)
          const hemispheres = key === 'stem' || key === 'cerebellum' ? [0] : [1, -1]
          for (const hemi of hemispheres) {
            const point = { x: anchor.x * (hemi || 1), y: anchor.y, z: anchor.z }
            const rotated = rotatePoint(point)
            const projected = project(rotated)
            const focused = !focusSector || focusSector === key
            const baseAlpha = focused ? 0.14 : 0.035
            const alpha = baseAlpha + baseAlpha * 0.4 * Math.sin(now * 0.0013 + hash(key) % 100)
            blit(sprite, projected.x, projected.y, Math.min(width, height) * 0.62, alpha)
          }
        }
        context.globalCompositeOperation = 'source-over'
      }

      function edgeControl(a, b, offset) {
        const dx = b.x - a.x
        const dy = b.y - a.y
        const length = Math.hypot(dx, dy) || 1
        return {
          x: (a.x + b.x) / 2 - (dy / length) * offset,
          y: (a.y + b.y) / 2 + (dx / length) * offset,
        }
      }

      function drawEdges(graph, now) {
        if (!graph.edges.length) return
        const groups = new Map()
        const hasSelection = !!(selectedId && graph.byId.has(selectedId))
        let index = 0
        for (const edge of graph.edges) {
          index += 1
          const from = graph.byId.get(edge.a)
          const to = graph.byId.get(edge.b)
          if (!from || !to) continue
          const a = project(rotatePoint(from.p))
          const b = project(rotatePoint(to.p))
          if (a.x < -60 || a.x > width + 60 || b.x < -60 || b.x > width + 60) continue
          const color = nodeColor(to)
          const kindAlpha = edge.kind === 'parent' ? 0.3 : edge.kind === 'anchor' ? 0.22 : edge.kind === 'time' ? 0.1 : edge.kind === 'cluster' ? 0.13 : 0.17
          const sectorDim = hasSelection
            ? 1
            : focusSector && from.sector !== focusSector && to.sector !== focusSector ? 0.2 : 1
          const connected = hasSelection && (edge.a === selectedId || edge.b === selectedId)
          const selectionDim = hasSelection ? (connected ? 1.7 : 0.18) : 1
          const classDim = classFilter && to.class !== classFilter ? (connected ? 1 : 0.14) : 1
          const alpha = kindAlpha * sectorDim * selectionDim * classDim
          if (alpha < 0.02) continue
          const offset = (hash(edge.a + edge.b + index) % 1000) / 1000 * 26 - 13
          const control = edgeControl(a, b, offset)
          const bucket = alpha > 0.26 ? 'strong' : alpha > 0.09 ? 'soft' : 'faint'
          const key = color + '|' + bucket
          let group = groups.get(key)
          if (!group) {
            group = { color, alpha: bucket === 'strong' ? 0.34 : bucket === 'soft' ? 0.15 : 0.06, paths: [] }
            groups.set(key, group)
          }
          group.paths.push(a.x, a.y, control.x, control.y, b.x, b.y)
        }
        context.globalCompositeOperation = 'lighter'
        context.lineWidth = 1
        for (const group of groups.values()) {
          context.strokeStyle = group.color
          context.globalAlpha = group.alpha
          context.beginPath()
          const paths = group.paths
          for (let cursor = 0; cursor < paths.length; cursor += 6) {
            context.moveTo(paths[cursor], paths[cursor + 1])
            context.quadraticCurveTo(paths[cursor + 2], paths[cursor + 3], paths[cursor + 4], paths[cursor + 5])
          }
          context.stroke()
        }
        context.globalAlpha = 1
        context.globalCompositeOperation = 'source-over'
      }

      function drawPulses(graph, now) {
        pulses = pulses.filter(pulse => now - pulse.at < pulse.duration)
        if (!pulses.length) return
        context.globalCompositeOperation = 'lighter'
        for (const pulse of pulses) {
          const from = graph.byId.get(pulse.from)
          const to = graph.byId.get(pulse.to)
          if (!to) continue
          const a = from ? project(rotatePoint(from.p)) : project(rotatePoint(to.p))
          const b = project(rotatePoint(to.p))
          const control = edgeControl(a, b, 12)
          const t = clamp((now - pulse.at) / pulse.duration, 0, 1)
          const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
          const x = (1 - eased) * (1 - eased) * a.x + 2 * (1 - eased) * eased * control.x + eased * eased * b.x
          const y = (1 - eased) * (1 - eased) * a.y + 2 * (1 - eased) * eased * control.y + eased * eased * b.y
          const alpha = Math.sin(t * Math.PI) * 0.85
          blit(whiteSprite, x, y, 26 * b.k, alpha)
          blit(sprites.get(pulse.sector) || whiteSprite, x, y, 46 * b.k, alpha * 0.5)
        }
        context.globalAlpha = 1
        context.globalCompositeOperation = 'source-over'
      }

      function drawNodes(graph, now, ts) {
        hits = []
        const ordered = []
        for (const node of graph.nodes) {
          const rotated = rotatePoint(node.p)
          const projected = project(rotated)
          ordered.push({ node, rotated, projected })
        }
        ordered.sort((a, b) => a.rotated.z - b.rotated.z)

        const hasSelection = !!(selectedId && graph.byId.has(selectedId))
        const selectedNeighbors = new Set()
        if (hasSelection) {
          for (const id of (graph.adjIn.get(selectedId) || [])) selectedNeighbors.add(id)
          for (const id of (graph.adjOut.get(selectedId) || [])) selectedNeighbors.add(id)
        }

        context.globalCompositeOperation = 'lighter'
        const labels = []
        for (const item of ordered) {
          const node = item.node
          const projected = item.projected
          if (projected.x < -60 || projected.x > width + 60 || projected.y < -60 || projected.y > height + 60) continue
          const focused = !focusSector || focusSector === node.sector
          const selected = node.id === selectedId
          const hovered = node.id === hoverId
          const neighbor = selectedNeighbors.has(node.id)
          const selectionDim = hasSelection ? (selected ? 1 : neighbor ? 0.72 : 0.22) : 1
          const classDim = !classFilter || node.class === classFilter
            ? 1
            : selected ? 1 : neighbor ? 0.7 : 0.16
          const dim = (hasSelection ? selectionDim : (focused ? 1 : 0.18)) * classDim
          const radius = Math.max(3.4, (4 + node.weight * 3.1) * projected.k * (0.9 + 0.1 * zoom)) * (selected ? 1.32 : 1)
          const age = now - node.updatedAt
          const pulse = !reducedMotion && age < 1000 ? 1 + 0.55 * Math.sin((1 - age / 1000) * Math.PI) : 1
          const family = node.family === 'skill' || node.family === 'mcp' ? node.family : null
          const color = nodeColor(node)
          const sprite = sprites.get(node.sector) || whiteSprite

          blit(sprite, projected.x, projected.y, radius * 7.4 * pulse, 0.5 * dim)
          if (selected) {
            blit(whiteSprite, projected.x, projected.y, radius * 15, 0.22)
            blit(sprite, projected.x, projected.y, radius * 9 * pulse, 0.55)
          } else if (hovered) {
            blit(whiteSprite, projected.x, projected.y, radius * 9, 0.4)
          }
          if (node.class === 'prompt') {
            const spread = radius + 6
            const accent = selected || hovered || classFilter === 'prompt'
            const diamondColor = accent ? CLASSES.prompt.color : color
            const diamondAlpha = accent ? 0.9 : 0.42
            context.fillStyle = hexToRgba(diamondColor, (accent ? 0.13 : 0.06) * dim)
            context.strokeStyle = hexToRgba(diamondColor, diamondAlpha * dim)
            context.lineWidth = selected ? 2.2 : accent ? 1.5 : 1.1
            context.beginPath()
            context.moveTo(projected.x, projected.y - spread)
            context.lineTo(projected.x + spread, projected.y)
            context.lineTo(projected.x, projected.y + spread)
            context.lineTo(projected.x - spread, projected.y)
            context.closePath()
            context.fill()
            context.stroke()
          }
          if (node.id === graph.lastActive && node.class !== 'prompt') {
            const ringRadius = radius + 7 + 2.5 * Math.sin(ts * 0.004)
            context.strokeStyle = hexToRgba(color, 0.55 * dim)
            context.lineWidth = 1.2
            context.beginPath()
            context.arc(projected.x, projected.y, ringRadius, 0, TAU)
            context.stroke()
          }

          context.globalAlpha = dim
          context.fillStyle = color
          context.beginPath()
          context.arc(projected.x, projected.y, radius, 0, TAU)
          context.fill()
          context.fillStyle = 'rgba(255,255,255,0.94)'
          context.beginPath()
          context.arc(projected.x, projected.y, Math.max(1.1, radius * 0.4), 0, TAU)
          context.fill()
          context.globalAlpha = 1

          if (family) {
            drawCallGlyph(family, projected.x, projected.y, radius, color, Math.max(dim, selected ? 1 : 0.35), selected)
          }

          if (node.status === 'error') {
            context.strokeStyle = 'rgba(255,92,92,0.9)'
            context.lineWidth = 1.5
            context.beginPath()
            context.arc(projected.x, projected.y, radius + 3.2, 0, TAU)
            context.stroke()
          } else if (node.status === 'running') {
            context.strokeStyle = hexToRgba(color, 0.85)
            context.lineWidth = 1.4
            context.setLineDash([3, 4])
            context.lineDashOffset = -ts * 0.012
            context.beginPath()
            context.arc(projected.x, projected.y, radius + 3.4, 0, TAU)
            context.stroke()
            context.setLineDash([])
            context.lineDashOffset = 0
          }

          if (selected) {
            context.strokeStyle = 'rgba(240,253,255,0.96)'
            context.lineWidth = 2
            context.beginPath()
            context.arc(projected.x, projected.y, radius + 7, 0, TAU)
            context.stroke()
            context.strokeStyle = hexToRgba(color, 0.9)
            context.lineWidth = 3.5
            context.setLineDash([4, 7])
            context.lineDashOffset = -ts * 0.014
            context.beginPath()
            context.arc(projected.x, projected.y, radius + 13, 0, TAU)
            context.stroke()
            context.setLineDash([])
            context.lineDashOffset = 0
            context.strokeStyle = 'rgba(235,250,255,0.8)'
            context.lineWidth = 1.2
            for (const angle of [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]) {
              const inner = radius + 17
              const outer = radius + 23
              context.beginPath()
              context.moveTo(projected.x + Math.cos(angle) * inner, projected.y + Math.sin(angle) * inner)
              context.lineTo(projected.x + Math.cos(angle) * outer, projected.y + Math.sin(angle) * outer)
              context.stroke()
            }
          }

          hits.push({ id: node.id, x: projected.x, y: projected.y, r: radius, k: projected.k })

          const important = selected
            || !!family
            || (classFilter === 'prompt' && node.class === 'prompt')
            || (graph.projectMode && (node.running || node.id === 'project'))
            || node.id === 'model'
          if (showLabels && important && labels.length < (family ? 6 : 5)) {
            labels.push({ node, x: projected.x, y: projected.y, radius, dim: Math.max(dim, selected ? 1 : 0.35), selected })
          }
        }
        context.globalCompositeOperation = 'source-over'

        for (const label of labels) {
          const text = String(opts.labelOf ? opts.labelOf(label.node) : label.node.title || label.node.toolName || '')
          if (!text) continue
          const trimmed = clip(text, label.selected ? 46 : 34)
          context.font = (label.selected ? '650 12px' : '600 11px') + ' "SF Pro Text", system-ui, -apple-system, "Segoe UI", sans-serif'
          const textWidth = context.measureText(trimmed).width
          const padding = 7
          const boxWidth = textWidth + padding * 2
          const boxHeight = label.selected ? 22 : 19
          let x = label.x - boxWidth / 2
          let y = label.y - label.radius - 16 - boxHeight
          x = clamp(x, 6, width - boxWidth - 6)
          y = clamp(y, 6, height - boxHeight - 6)
          context.globalAlpha = (label.selected ? 1 : 0.86) * label.dim
          roundRectPath(x, y, boxWidth, boxHeight, label.selected ? 8 : 7)
          context.fillStyle = label.selected ? 'rgba(5,14,26,0.94)' : 'rgba(4,9,18,0.82)'
          context.fill()
          context.strokeStyle = hexToRgba(nodeColor(label.node), (label.selected ? 0.95 : 0.45) * label.dim)
          context.lineWidth = label.selected ? 1.6 : 1
          context.stroke()
          context.fillStyle = label.selected ? '#ffffff' : 'rgba(234,246,255,0.94)'
          context.textBaseline = 'middle'
          context.fillText(trimmed, x + padding, y + boxHeight / 2 + 0.5)
          context.globalAlpha = 1
        }
      }

      function consumeBirths(graph) {
        if (!graph) return
        if (graph.epoch !== epoch) {
          epoch = graph.epoch
          birthCursor = graph.births.length
          return
        }
        while (birthCursor < graph.births.length) {
          const birth = graph.births[birthCursor]
          birthCursor += 1
          if (!birth || Date.now() - birth.at > 6000) continue
          if (pulses.length > 40) pulses.shift()
          pulses.push({
            from: birth.parent || null,
            to: birth.id,
            sector: (graph.byId.get(birth.id) || {}).sector || 'parietal',
            at: performance.now(),
            duration: 1100,
          })
        }
      }

      function frame(ts) {
        if (destroyed) return
        raf = window.requestAnimationFrame(frame)
        if (document.hidden) return
        /* Cap drawing at ~75 fps so a 120 Hz display does not double the work. */
        if (ts - lastFrame < 12) return
        lastFrame = ts
        const now = performance.now()
        const graph = graphSource()
        if (!graph) {
          context.clearRect(0, 0, width, height)
          return
        }
        consumeBirths(graph)

        if (!dragging && !hoverId && now > holdUntil) {
          if (!reducedMotion) targetYaw += 0.0017
        }
        const yawDelta = Math.atan2(Math.sin(targetYaw - yaw), Math.cos(targetYaw - yaw))
        yaw += yawDelta * 0.09
        pitch += (targetPitch - pitch) * 0.09
        zoom += (targetZoom - zoom) * 0.12

        context.setTransform(dpr, 0, 0, dpr, 0, 0)
        context.clearRect(0, 0, width, height)

        drawCore(now)
        drawShell(now)
        context.globalCompositeOperation = 'lighter'
        drawCloud(now)
        context.globalCompositeOperation = 'source-over'
        drawRings(now)
        drawAmbient(now)
        drawEdges(graph, now)
        drawSectorHalos(now)
        drawNodes(graph, now, ts)
        drawPulses(graph, now)
      }

      function hitTest(x, y) {
        let best = null
        let bestScore = -Infinity
        for (const hit of hits) {
          const distance = Math.hypot(hit.x - x, hit.y - y)
          if (distance > Math.max(10, hit.r + 8)) continue
          const score = hit.k * 1000 - distance
          if (score > bestScore) { bestScore = score; best = hit }
        }
        return best
      }

      function onPointerDown(event) {
        if (event.button !== 0) return
        if (opts.onHover) opts.onHover(null)
        dragging = true
        pointerId = event.pointerId
        downX = event.clientX
        downY = event.clientY
        lastX = event.clientX
        lastY = event.clientY
        moved = 0
        downAt = performance.now()
        holdUntil = downAt + 4000
        canvas.setPointerCapture(event.pointerId)
        canvas.style.cursor = 'grabbing'
      }

      function onPointerMove(event) {
        if (dragging && event.pointerId === pointerId) {
          const dx = event.clientX - lastX
          const dy = event.clientY - lastY
          lastX = event.clientX
          lastY = event.clientY
          moved += Math.abs(dx) + Math.abs(dy)
          targetYaw += dx * 0.0062
          yaw += dx * 0.0062
          targetPitch = clamp(targetPitch + dy * 0.0042, -0.7, 0.7)
          pitch = clamp(pitch + dy * 0.0042, -0.7, 0.7)
          holdUntil = performance.now() + 3500
          return
        }
        const rect = canvas.getBoundingClientRect()
        const hit = hitTest(event.clientX - rect.left, event.clientY - rect.top)
        const nextHover = hit ? hit.id : null
        if (nextHover !== hoverId) {
          hoverId = nextHover
          canvas.style.cursor = nextHover ? 'pointer' : 'grab'
          if (opts.onHover) {
            opts.onHover(hit ? { id: hit.id, x: hit.x, y: hit.y } : null)
          }
        }
        if (nextHover) holdUntil = Math.max(holdUntil, performance.now() + 1800)
      }

      function onPointerUp(event) {
        if (!dragging) return
        dragging = false
        const elapsed = performance.now() - downAt
        if (moved < 6 && elapsed < 700) {
          const rect = canvas.getBoundingClientRect()
          const hit = hitTest(event.clientX - rect.left, event.clientY - rect.top)
          if (opts.onSelect) opts.onSelect(hit ? hit.id : null)
        }
        try { canvas.releasePointerCapture(event.pointerId) } catch {}
        pointerId = null
        canvas.style.cursor = hoverId ? 'pointer' : 'grab'
      }

      function onPointerLeave() {
        if (dragging) return
        if (hoverId !== null) {
          hoverId = null
          canvas.style.cursor = 'grab'
          if (opts.onHover) opts.onHover(null)
        }
      }

      function onWheel(event) {
        event.preventDefault()
        const delta = Math.sign(event.deltaY) * Math.min(120, Math.abs(event.deltaY))
        targetZoom = clamp(targetZoom * Math.exp(-delta * 0.0016), 0.55, 2.7)
        holdUntil = performance.now() + 3500
      }

      function onDoubleClick(event) {
        if (event && opts.onActivate) {
          const rect = canvas.getBoundingClientRect()
          const hit = hitTest(event.clientX - rect.left, event.clientY - rect.top)
          if (hit) {
            opts.onActivate(hit.id)
            return
          }
        }
        targetYaw = 0.45
        targetPitch = -0.1
        targetZoom = 1
        holdUntil = 0
      }

      function cycleNode(direction) {
        const graph = graphSource()
        if (!graph || !graph.nodes.length) return
        const ordered = graph.nodes.slice().sort((a, b) => a.time - b.time)
        let index = selectedId ? ordered.findIndex(node => node.id === selectedId) : -1
        if (index < 0) index = direction > 0 ? -1 : 0
        index = (index + direction + ordered.length) % ordered.length
        const node = ordered[index]
        if (!node) return
        selectedId = node.id
        if (opts.onSelect) opts.onSelect(node.id)
        holdUntil = performance.now() + 2200
      }

      function onKeyDown(event) {
        if (event.key === 'ArrowLeft') { targetYaw -= 0.14; event.preventDefault() }
        else if (event.key === 'ArrowRight') { targetYaw += 0.14; event.preventDefault() }
        else if (event.key === 'ArrowUp') { targetPitch = clamp(targetPitch - 0.1, -0.7, 0.7); event.preventDefault() }
        else if (event.key === 'ArrowDown') { targetPitch = clamp(targetPitch + 0.1, -0.7, 0.7); event.preventDefault() }
        else if (event.key === '+' || event.key === '=') { targetZoom = clamp(targetZoom * 1.15, 0.55, 2.7); event.preventDefault() }
        else if (event.key === '-') { targetZoom = clamp(targetZoom / 1.15, 0.55, 2.7); event.preventDefault() }
        else if (event.key === '0') { onDoubleClick(); event.preventDefault() }
        else if (event.key === ']' || event.code === 'BracketRight' || event.key === 'PageDown') { cycleNode(1); event.preventDefault() }
        else if (event.key === '[' || event.code === 'BracketLeft' || event.key === 'PageUp') { cycleNode(-1); event.preventDefault() }
        else if (event.key === 'Enter') {
          if (selectedId && opts.onSelect) opts.onSelect(selectedId)
          event.preventDefault()
        }
        if (event.key.startsWith('Arrow') || event.key === '+' || event.key === '-') {
          holdUntil = performance.now() + 2500
        }
      }

      canvas.addEventListener('pointerdown', onPointerDown)
      canvas.addEventListener('pointermove', onPointerMove)
      canvas.addEventListener('pointerup', onPointerUp)
      canvas.addEventListener('pointercancel', onPointerUp)
      canvas.addEventListener('pointerleave', onPointerLeave)
      canvas.addEventListener('wheel', onWheel, { passive: false })
      canvas.addEventListener('dblclick', onDoubleClick)
      canvas.addEventListener('keydown', onKeyDown)
      canvas.style.cursor = 'grab'

      function start() {
        if (raf) return
        raf = window.requestAnimationFrame(frame)
      }

      function stop() {
        if (raf) window.cancelAnimationFrame(raf)
        raf = 0
      }

      function focusNode(id) {
        const graph = graphSource()
        if (!graph) return
        const node = graph.byId.get(id)
        if (!node) return
        targetYaw = Math.atan2(-node.p.x, node.p.z)
        targetPitch = clamp(node.p.y * 0.6, -0.55, 0.55)
        targetZoom = Math.max(targetZoom, 1.45)
        holdUntil = performance.now() + 5000
      }

      function destroy() {
        destroyed = true
        stop()
        canvas.removeEventListener('pointerdown', onPointerDown)
        canvas.removeEventListener('pointermove', onPointerMove)
        canvas.removeEventListener('pointerup', onPointerUp)
        canvas.removeEventListener('pointercancel', onPointerUp)
        canvas.removeEventListener('pointerleave', onPointerLeave)
        canvas.removeEventListener('wheel', onWheel)
        canvas.removeEventListener('dblclick', onDoubleClick)
        canvas.removeEventListener('keydown', onKeyDown)
      }

      return {
        start,
        stop,
        resize,
        destroy,
        focusNode,
        setGraphSource(source) { graphSource = source },
        setSelection(id) { selectedId = id || null },
        setFocusSector(key) { focusSector = key || null },
        setClassFilter(key) { classFilter = key || null },
        setLabels(value) { showLabels = value !== false },
        setAutoLabels(value) { showLabels = value !== false },
        zoomBy(factor) {
          targetZoom = clamp(targetZoom * factor, 0.55, 2.7)
          holdUntil = performance.now() + 2500
        },
        resetView() {
          targetYaw = 0.45
          targetPitch = -0.1
          targetZoom = 1
          holdUntil = 0
        },
      }
    }

    /* ------------------------------------------------------------------ *
     *  UI store
     * ------------------------------------------------------------------ */

    const SIDE_DEFAULT = 440
    const SIDE_MIN = 340
    const SIDE_MAX = 640
    const SIDE_KEY = 'dsh.neural-thoughts.sideWidth'

    function sideWidthValue(value) {
      const number = Number(value)
      if (!Number.isFinite(number)) return SIDE_DEFAULT
      return Math.max(SIDE_MIN, Math.min(SIDE_MAX, Math.round(number)))
    }

    function readSideWidth() {
      try {
        const stored = window.localStorage.getItem(SIDE_KEY)
        return sideWidthValue(stored === null ? SIDE_DEFAULT : stored)
      } catch { return SIDE_DEFAULT }
    }

    function saveSideWidth(value) {
      try { window.localStorage.setItem(SIDE_KEY, String(sideWidthValue(value))) } catch {}
    }

    function createUiStore() {
      let state = {
        open: false,
        mode: 'chat',
        sessionId: null,
        workspaceId: null,
        tab: 'flow',
        selectedId: null,
        sector: null,
        classFilter: null,
        mediaId: null,
        fileId: null,
        filter: '',
        fullscreen: false,
        labels: true,
        flowLimit: 40,
        sideWidth: readSideWidth(),
        version: 0,
      }
      const listeners = new Set()
      return {
        getSnapshot() { return state },
        subscribe(listener) {
          listeners.add(listener)
          return () => { listeners.delete(listener) }
        },
        patch(patch) {
          const next = Object.assign({}, state, patch)
          let changed = false
          for (const key of Object.keys(next)) {
            if (next[key] !== state[key]) { changed = true; break }
          }
          if (!changed) return
          next.version = state.version + 1
          state = next
          for (const listener of [...listeners]) listener()
        },
      }
    }

    /* ------------------------------------------------------------------ *
     *  Small icons
     * ------------------------------------------------------------------ */

    function BrainMark(props) {
      const size = props && props.size ? props.size : 22
      const accent = props && props.accent ? props.accent : '#5fe3ff'
      return h('svg', {
        width: size, height: size, viewBox: '0 0 32 32', 'aria-hidden': 'true', focusable: 'false',
      },
        h('g', { fill: 'none', stroke: accent, strokeWidth: 1.15, strokeLinecap: 'round', opacity: 0.85 },
          h('path', { d: 'M9.6 9.2c-2.4 2.1-3.2 5.6-1.8 8.5 1.2 2.5 3.9 3.9 6.6 3.5' }),
          h('path', { d: 'M22.2 8.7c2.6 2.3 3.3 6.1 1.7 9.1-1.3 2.4-4 3.8-6.7 3.4' }),
          h('path', { d: 'M11 8.9c1.5-1.5 4-1.9 5.9-.9 1.6.9 2.5 2.6 2.4 4.4' }),
          h('path', { d: 'M16.1 21.3c1.7 1 3.8.7 5.1-.8' }),
          h('path', { d: 'M15.7 12.4c-1.4-.4-2.9.3-3.5 1.6-.6 1.2-.2 2.7.8 3.6' }),
          h('path', { d: 'M15.7 12.4c1.6-.1 3 .8 3.6 2.2.5 1.3.1 2.7-.9 3.6' }),
          h('path', { d: 'M12.9 17.7c.6 1.6 2.2 2.5 3.8 2.2' })
        ),
        h('g', { fill: '#ffffff' },
          h('circle', { cx: 9.6, cy: 9.2, r: 1.25 }),
          h('circle', { cx: 22.2, cy: 8.7, r: 1.25 }),
          h('circle', { cx: 16, cy: 12.4, r: 1.35 }),
          h('circle', { cx: 12.1, cy: 17.9, r: 1.1 }),
          h('circle', { cx: 18.9, cy: 18.1, r: 1.1 }),
          h('circle', { cx: 16.6, cy: 21.9, r: 1.2 })
        ),
        h('g', { fill: accent },
          h('circle', { cx: 16, cy: 12.4, r: 0.62 }),
          h('circle', { cx: 16.6, cy: 21.9, r: 0.55 })
        )
      )
    }

    function IconChevron() {
      return h('svg', { width: 12, height: 12, viewBox: '0 0 12 12', 'aria-hidden': 'true' },
        h('path', { d: 'M3 4.5 6 7.5 9 4.5', fill: 'none', stroke: 'currentColor', strokeWidth: 1.4, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconArrowRight() {
      return h('svg', { width: 12, height: 12, viewBox: '0 0 12 12', 'aria-hidden': 'true' },
        h('path', { d: 'M2.2 6h7.2M6.7 3.2 9.5 6l-2.8 2.8', fill: 'none', stroke: 'currentColor', strokeWidth: 1.3, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconClose() {
      return h('svg', { width: 14, height: 14, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M3.5 3.5 10.5 10.5M10.5 3.5 3.5 10.5', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round' }))
    }

    function IconRefresh() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M11.4 6.1A4.5 4.5 0 1 0 10 10', fill: 'none', stroke: 'currentColor', strokeWidth: 1.3, strokeLinecap: 'round' }),
        h('path', { d: 'M11.6 2.7v3.6H8', fill: 'none', stroke: 'currentColor', strokeWidth: 1.3, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconSearch() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('circle', { cx: 6, cy: 6, r: 3.7, fill: 'none', stroke: 'currentColor', strokeWidth: 1.3 }),
        h('path', { d: 'M8.9 8.9 12 12', fill: 'none', stroke: 'currentColor', strokeWidth: 1.3, strokeLinecap: 'round' }))
    }

    function IconPlus(props) {
      const size = props && props.size ? props.size : 13
      return h('svg', { width: size, height: size, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M7 2.6v8.8M2.6 7h8.8', fill: 'none', stroke: 'currentColor', strokeWidth: 1.3, strokeLinecap: 'round' }))
    }

    function IconMinus(props) {
      const size = props && props.size ? props.size : 13
      return h('svg', { width: size, height: size, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M2.6 7h8.8', fill: 'none', stroke: 'currentColor', strokeWidth: 1.3, strokeLinecap: 'round' }))
    }

    function IconOrbit() {
      return h('svg', { width: 14, height: 14, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('circle', { cx: 7, cy: 7, r: 2.1, fill: 'none', stroke: 'currentColor', strokeWidth: 1.2 }),
        h('ellipse', { cx: 7, cy: 7, rx: 5.6, ry: 2.6, fill: 'none', stroke: 'currentColor', strokeWidth: 1.1, opacity: 0.7, transform: 'rotate(-24 7 7)' }))
    }

    function IconBolt() {
      return h('svg', { width: 12, height: 12, viewBox: '0 0 12 12', 'aria-hidden': 'true' },
        h('path', { d: 'M6.7 1.2 2.9 6.5h2.3L5.1 10.8 9 5.4H6.6l.1-4.2Z', fill: 'currentColor' }))
    }

    function IconCheck() {
      return h('svg', { width: 12, height: 12, viewBox: '0 0 12 12', 'aria-hidden': 'true' },
        h('path', { d: 'M2.4 6.3 4.8 8.7 9.7 3.6', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconAlert() {
      return h('svg', { width: 12, height: 12, viewBox: '0 0 12 12', 'aria-hidden': 'true' },
        h('path', { d: 'M6 2.2v4.4M6 9.4v.2', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round' }))
    }

    function IconSpinner() {
      return h('svg', { className: 'dsh-nt-spinner', width: 14, height: 14, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('circle', { cx: 7, cy: 7, r: 5.2, fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, opacity: 0.25 }),
        h('path', { d: 'M7 1.8a5.2 5.2 0 0 1 5.2 5.2', fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round' }))
    }

    function IconGlobe() {
      return h('svg', { width: 14, height: 14, viewBox: '0 0 16 16', 'aria-hidden': 'true' },
        h('circle', { cx: 8, cy: 8, r: 5.6, fill: 'none', stroke: 'currentColor', strokeWidth: 1.2 }),
        h('ellipse', { cx: 8, cy: 8, rx: 2.5, ry: 5.6, fill: 'none', stroke: 'currentColor', strokeWidth: 1, opacity: 0.7 }),
        h('path', { d: 'M2.6 8h10.8M3.4 5.2h9.2M3.4 10.8h9.2', fill: 'none', stroke: 'currentColor', strokeWidth: 1, opacity: 0.7 }))
    }

    function IconFullscreen() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M5.2 2.2H2.2v3M8.8 2.2h3v3M5.2 11.8H2.2v-3M8.8 11.8h3v-3', fill: 'none', stroke: 'currentColor', strokeWidth: 1.3, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconHelp() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('circle', { cx: 7, cy: 7, r: 5.3, fill: 'none', stroke: 'currentColor', strokeWidth: 1.2 }),
        h('path', { d: 'M5.4 5.4c0-1 .7-1.8 1.7-1.8 1 0 1.7.7 1.7 1.6 0 .9-.6 1.2-1.1 1.7-.4.4-.6.7-.6 1.2', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round' }),
        h('circle', { cx: 7, cy: 10.2, r: 0.7, fill: 'currentColor' }))
    }

    function IconCopy() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('rect', { x: 4.6, y: 4.6, width: 7, height: 7, rx: 1.4, fill: 'none', stroke: 'currentColor', strokeWidth: 1.2 }),
        h('path', { d: 'M9.4 4.6V3.4A1.4 1.4 0 0 0 8 2H3.4A1.4 1.4 0 0 0 2 3.4V8a1.4 1.4 0 0 0 1.4 1.4h1.2', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round' }))
    }

    function IconThought() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M7 1.6 8.2 5.8 12.4 7 8.2 8.2 7 12.4 5.8 8.2 1.6 7 5.8 5.8Z', fill: 'currentColor' }),
        h('circle', { cx: 11.6, cy: 2.4, r: 1, fill: 'currentColor', opacity: 0.7 }),
        h('circle', { cx: 2.4, cy: 11.6, r: 0.8, fill: 'currentColor', opacity: 0.55 }))
    }

    function IconPrompt() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M2.4 2.8h9.2v6.1H7.4L4.2 11.8V8.9H2.4z', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinejoin: 'round' }),
        h('path', { d: 'M4.6 5.2h4.8M4.6 7h3.2', fill: 'none', stroke: 'currentColor', strokeWidth: 1.1, strokeLinecap: 'round' }))
    }

    function IconDoc() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M3.4 1.8h4.4l2.8 2.8v7.6H3.4z', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinejoin: 'round' }),
        h('path', { d: 'M7.8 1.8v2.8h2.8M5.2 8h3.6M5.2 10.2h2.4', fill: 'none', stroke: 'currentColor', strokeWidth: 1.1, strokeLinecap: 'round' }))
    }

    function IconFlag() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M4 12.2V2M4 2.6h6.2L8.8 5l1.4 2.4H4', fill: 'none', stroke: 'currentColor', strokeWidth: 1.25, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconSkill() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M7 1.7 11.6 4.3v5.4L7 12.3 2.4 9.7V4.3z', fill: 'none', stroke: 'currentColor', strokeWidth: 1.15, strokeLinejoin: 'round' }),
        h('path', { d: 'M7 4.3v5.4M4.7 5.6l4.6 2.8M9.3 5.6 4.7 8.4', fill: 'none', stroke: 'currentColor', strokeWidth: 0.95, strokeLinecap: 'round' }))
    }

    function IconMcp() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('rect', { x: 2.2, y: 4.1, width: 6.2, height: 5.8, rx: 1.3, fill: 'none', stroke: 'currentColor', strokeWidth: 1.15 }),
        h('path', { d: 'M8.4 5.6h2.1v2.8H8.4M10.5 7H12M5.2 9.9v2.2M4 12.1h2.4', fill: 'none', stroke: 'currentColor', strokeWidth: 1.15, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconAgent() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('circle', { cx: 7, cy: 4.6, r: 2.1, fill: 'none', stroke: 'currentColor', strokeWidth: 1.2 }),
        h('path', { d: 'M2.9 12c.2-2.3 1.9-3.7 4.1-3.7s3.9 1.4 4.1 3.7', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round' }))
    }

    function EventIcon(props) {
      const kind = props && props.kind
      if (kind === 'prompt') return h(IconPrompt)
      if (kind === 'reasoning' || kind === 'answer' || kind === 'attempt') return h(IconThought)
      if (kind === 'document' || kind === 'reference') return h(IconDoc)
      if (kind === 'plan' || kind === 'goal' || kind === 'deliverable') return h(IconFlag)
      if (kind === 'agent') return h(IconAgent)
      if (kind === 'error') return h(IconAlert)
      return h(IconBolt)
    }

    function CallFamilyIcon(props) {
      return props && props.family === 'skill' ? h(IconSkill) : h(IconMcp)
    }

    function IconPlay() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M4.2 2.4 11.6 7l-7.4 4.6Z', fill: 'currentColor' }))
    }

    function IconPause() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('rect', { x: 3.6, y: 2.4, width: 2.4, height: 9.2, rx: 0.8, fill: 'currentColor' }),
        h('rect', { x: 8, y: 2.4, width: 2.4, height: 9.2, rx: 0.8, fill: 'currentColor' }))
    }

    function IconRestart() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M2.5 6.2A4.8 4.8 0 1 1 2.8 9M2.5 2.8v3.6h3.6', fill: 'none', stroke: 'currentColor', strokeWidth: 1.25, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconStepBack() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M3.1 2.7v8.6M10.8 3.1 5.3 7l5.5 3.9Z', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconStepForward() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M10.9 2.7v8.6M3.2 3.1 8.7 7l-5.5 3.9Z', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconLoop() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M2.4 5.7A4.2 4.2 0 0 1 6.6 2.1h3.2M11.6 8.3a4.2 4.2 0 0 1-4.2 3.6H4.2M10.8 1.4l2 1.8-2 1.8M3.2 12.6l-2-1.8 2-1.8', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconHistory() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M2.6 6.4A4.5 4.5 0 1 1 3 9.2M2.4 2.9v3.7h3.7', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round', strokeLinejoin: 'round' }),
        h('path', { d: 'M7 4.2v3.3l2.1 1.2', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconLayers() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M7 1.8 12.2 4.4 7 7 1.8 4.4Z', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinejoin: 'round' }),
        h('path', { d: 'M1.8 7 7 9.6 12.2 7M1.8 9.6 7 12.2 12.2 9.6', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconImage() {
      return h('svg', { width: 14, height: 14, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('rect', { x: 1.8, y: 2.4, width: 10.4, height: 9.2, rx: 1.6, fill: 'none', stroke: 'currentColor', strokeWidth: 1.2 }),
        h('circle', { cx: 5, cy: 5.6, r: 1.1, fill: 'currentColor' }),
        h('path', { d: 'm3.2 10 2.7-2.8 1.8 1.6 1.6-1.4 1.5 2.6', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconVideo() {
      return h('svg', { width: 14, height: 14, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('rect', { x: 1.6, y: 3, width: 10.8, height: 8, rx: 1.7, fill: 'none', stroke: 'currentColor', strokeWidth: 1.2 }),
        h('path', { d: 'M5.8 5.1 9.2 7l-3.4 1.9Z', fill: 'currentColor' }))
    }

    function IconDownload() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M7 2.2v6.2M4.4 6.1 7 8.7l2.6-2.6M2.4 11.4h9.2', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round', strokeLinejoin: 'round' }))
    }

    function IconLabel() {
      return h('svg', { width: 13, height: 13, viewBox: '0 0 14 14', 'aria-hidden': 'true' },
        h('path', { d: 'M2.4 3.8h7.2M2.4 7h9.2M2.4 10.2h5.4', fill: 'none', stroke: 'currentColor', strokeWidth: 1.2, strokeLinecap: 'round' }),
        h('circle', { cx: 11.2, cy: 10.2, r: 1, fill: 'currentColor' }))
    }

    /* ------------------------------------------------------------------ *
     *  Hooks and small helpers
     * ------------------------------------------------------------------ */

    function useUi(uiStore) {
      return useSyncExternalStore(uiStore.subscribe, uiStore.getSnapshot, uiStore.getSnapshot)
    }

    function useRuntime(runtime) {
      return useSyncExternalStore(runtime.subscribe, runtime.getSnapshot, runtime.getSnapshot)
    }

    function useSessionList(ctx) {
      const source = ctx.sessions.list
      const subscribe = useMemo(() => listener => source.subscribe(listener), [source])
      const getSnapshot = useMemo(() => () => source.getSnapshot(), [source])
      return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
    }

    function useLang() {
      return useSyncExternalStore(languageStore.subscribe, languageStore.getSnapshot, languageStore.getSnapshot)
    }

    function useT() {
      const lang = useLang()
      return useMemo(() => makeTranslate(lang), [lang])
    }

    function timeAgo(ms, locale, t) {
      const diff = Math.max(0, Date.now() - (Number(ms) || 0))
      if (!Number.isFinite(diff)) return ''
      try {
        const formatter = new Intl.RelativeTimeFormat(String(locale || 'en'), { numeric: 'auto' })
        if (diff < 45000) return formatter.format(-Math.round(diff / 1000), 'second')
        if (diff < 3600000) return formatter.format(-Math.round(diff / 60000), 'minute')
        if (diff < 86400000) return formatter.format(-Math.round(diff / 3600000), 'hour')
        return formatter.format(-Math.round(diff / 86400000), 'day')
      } catch {
        return t('time.justNow')
      }
    }

    function clockOf(ms, locale) {
      try {
        return new Date(Number(ms) || Date.now()).toLocaleTimeString(String(locale || 'en'), {
          hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
        })
      } catch { return '' }
    }

    function sectorLabel(key, t, short) {
      return t('sector.' + key + (short ? '.short' : ''))
    }

    function kindLabel(kind, t) {
      const key = 'node.kind.' + kind
      const value = t(key)
      return value === key ? t('node.unknown') : value
    }

    function statusLabel(status, t) {
      const key = 'node.status.' + status
      const value = t(key)
      return value === key ? status : value
    }

    function nodeTitle(node, t) {
      if (!node) return ''
      if (node.titleKey) return t(node.titleKey)
      if (node.title) return node.title
      if (node.toolName) return node.toolName
      return t('node.unknown')
    }

    /* ------------------------------------------------------------------ *
     *  Session picker
     * ------------------------------------------------------------------ */

    function LanguageMenu(props) {
      const { t } = props
      const lang = useLang()
      const [open, setOpen] = useState(false)
      const ref = useRef(null)
      const menuRef = useRef(null)
      const buttonRef = useRef(null)

      useEffect(() => {
        if (!open || !menuRef.current) return
        const items = menuRef.current.querySelectorAll('[role="menuitemradio"]')
        const active = Array.from(items).find(item => item.getAttribute('aria-checked') === 'true') || items[0]
        if (active) active.focus()
      }, [open])

      function moveMenuFocus(event) {
        if (!menuRef.current) return
        const items = Array.from(menuRef.current.querySelectorAll('[role="menuitemradio"]'))
        if (!items.length) return
        const index = items.indexOf(document.activeElement)
        let next = null
        if (event.key === 'ArrowDown') next = (index + 1 + items.length) % items.length
        else if (event.key === 'ArrowUp') next = (index - 1 + items.length) % items.length
        else if (event.key === 'Home') next = 0
        else if (event.key === 'End') next = items.length - 1
        if (next === null) return
        event.preventDefault()
        items[next].focus()
      }

      useEffect(() => {
        if (!open) return undefined
        const onPointerDown = event => {
          if (ref.current && !ref.current.contains(event.target)) setOpen(false)
        }
        const onKeyDown = event => {
          if (event.key === 'Escape') {
            event.stopImmediatePropagation()
            event.preventDefault()
            setOpen(false)
            if (buttonRef.current) buttonRef.current.focus()
          }
        }
        document.addEventListener('pointerdown', onPointerDown, true)
        document.addEventListener('keydown', onKeyDown, true)
        return () => {
          document.removeEventListener('pointerdown', onPointerDown, true)
          document.removeEventListener('keydown', onKeyDown, true)
        }
      }, [open])

      const current = LANGS.find(item => item.id === lang) || LANGS[1]

      return h('div', { className: 'dsh-nt-lang', ref },
        h('button', {
          ref: buttonRef,
          type: 'button',
          className: 'dsh-nt-lang-button',
          title: t('lang.label'),
          'aria-label': t('lang.label'),
          'aria-haspopup': 'menu',
          'aria-expanded': open ? 'true' : 'false',
          onClick: () => setOpen(value => !value),
        },
        h(IconGlobe),
        h('span', { className: 'dsh-nt-lang-code' }, lang.toUpperCase())
        ),
        open ? h('div', {
          className: 'dsh-nt-lang-pop',
          role: 'menu',
          ref: menuRef,
          'aria-label': t('lang.label'),
          onKeyDown: moveMenuFocus,
        },
          LANGS.map(item => h('button', {
            key: item.id,
            type: 'button',
            role: 'menuitemradio',
            'aria-checked': item.id === lang,
            className: 'dsh-nt-lang-item' + (item.id === lang ? ' is-active' : ''),
            onClick: () => {
              languageStore.set(item.id)
              setOpen(false)
              if (buttonRef.current) buttonRef.current.focus()
            },
          },
          h('span', { className: 'dsh-nt-lang-native' }, item.label),
          h('span', { className: 'dsh-nt-lang-id' }, item.id.toUpperCase()),
          item.id === lang ? h('span', { className: 'dsh-nt-lang-check', 'aria-hidden': 'true' }, h(IconCheck)) : null
          ))
        ) : null
      )
    }

    function ModeToggle(props) {
      const { mode, t, onChange, projectsAvailable } = props
      const items = [
        { id: 'chat', label: t('mode.chat'), disabled: false },
        { id: 'project', label: t('mode.project'), disabled: !projectsAvailable },
      ]
      return h('div', { className: 'dsh-nt-mode', role: 'group', 'aria-label': t('mode.chat') + ' / ' + t('mode.project') },
        items.map(item => h('button', {
          key: item.id,
          type: 'button',
          className: 'dsh-nt-mode-button' + (mode === item.id ? ' is-active' : ''),
          'aria-pressed': mode === item.id,
          disabled: item.disabled,
          title: item.disabled ? t('project.noProject') : item.label,
          onClick: () => { if (!item.disabled) onChange(item.id) },
        }, item.label))
      )
    }

    function SessionPicker(props) {
      const {
        mode = 'chat', list, selectedId, selectedTitle, t, locale, onPick,
        workspaces, selectedWorkspaceId, selectedWorkspaceTitle, onPickWorkspace,
        onRefresh, refreshing,
      } = props
      const projectMode = mode === 'project'
      const [open, setOpen] = useState(false)
      const [query, setQuery] = useState('')
      const wrapperRef = useRef(null)
      const inputRef = useRef(null)
      const buttonRef = useRef(null)

      const runningCount = useMemo(() => {
        const ids = list && Array.isArray(list.ids) ? list.ids : []
        const byId = list && list.byId ? list.byId : {}
        let count = 0
        for (const id of ids) {
          if (byId[id] && byId[id].running) count += 1
        }
        return count
      }, [list])

      const workspaceRows = useMemo(() => {
        if (!open || !projectMode) return []
        const byId = list && list.byId ? list.byId : {}
        const needle = query.trim().toLowerCase()
        return (workspaces && Array.isArray(workspaces.items) ? workspaces.items : [])
          .map(workspace => {
            const ids = (Array.isArray(workspace.sessionIds) ? workspace.sessionIds : [])
              .filter(id => byId[id] && !byId[id].blank)
            let running = 0
            for (const id of ids) {
              if (byId[id] && byId[id].running) running += 1
            }
            return {
              id: workspace.workspaceId,
              title: workspace.title,
              path: workspace.path,
              chatCount: ids.length,
              runningCount: running,
              updatedAt: Date.parse(workspace.updatedAt || '') || 0,
            }
          })
          .filter(workspace => {
            if (!needle) return true
            return String(workspace.title || '').toLowerCase().includes(needle)
              || String(workspace.path || '').toLowerCase().includes(needle)
          })
          .sort((a, b) => {
            if (!!a.runningCount !== !!b.runningCount) return a.runningCount ? -1 : 1
            return (b.updatedAt || 0) - (a.updatedAt || 0)
          })
      }, [open, projectMode, workspaces, query, list])

      const selectedRunningCount = useMemo(() => {
        if (!projectMode) return 0
        const byId = list && list.byId ? list.byId : {}
        const items = workspaces && Array.isArray(workspaces.items) ? workspaces.items : []
        const workspace = items.find(item => item.workspaceId === selectedWorkspaceId)
        if (!workspace) return 0
        let count = 0
        for (const id of (workspace.sessionIds || [])) {
          if (byId[id] && byId[id].running) count += 1
        }
        return count
      }, [projectMode, workspaces, selectedWorkspaceId, list])

      function closeAndReturnFocus() {
        setOpen(false)
        setQuery('')
        if (buttonRef.current) buttonRef.current.focus()
      }

      useEffect(() => {
        if (!open) return undefined
        const onPointerDown = event => {
          if (wrapperRef.current && !wrapperRef.current.contains(event.target)) setOpen(false)
        }
        document.addEventListener('pointerdown', onPointerDown, true)
        return () => document.removeEventListener('pointerdown', onPointerDown, true)
      }, [open])

      useEffect(() => {
        if (open && inputRef.current) inputRef.current.focus()
      }, [open])

      const rows = useMemo(() => {
        if (!open) return []
        const ids = list && Array.isArray(list.ids) ? list.ids.slice() : []
        const byId = list && list.byId ? list.byId : {}
        const seen = new Set()
        const collected = []
        for (const id of ids) {
          const row = byId[id]
          if (!row || seen.has(id)) continue
          seen.add(id)
          collected.push(row)
        }
        for (const id of Object.keys(byId)) {
          if (seen.has(id)) continue
          const row = byId[id]
          if (!row) continue
          if (row.retainedBy && row.retainedBy['neural-thoughts']) {
            collected.push(row)
            seen.add(id)
          }
        }
        return collected
          .filter(row => {
            if (!query.trim()) return true
            const needle = query.trim().toLowerCase()
            return String(row.displayTitle || row.title || '').toLowerCase().includes(needle)
              || String(row.cwd || '').toLowerCase().includes(needle)
          })
          .sort((a, b) => {
            if (!!a.running !== !!b.running) return a.running ? -1 : 1
            return (b.updatedAt || 0) - (a.updatedAt || 0)
          })
          .slice(0, 60)
      }, [list, query, open])

      const chatBody = rows.length
        ? rows.map(row => h('button', {
          key: row.id,
          type: 'button',
          className: 'dsh-nt-session-row' + (row.id === selectedId ? ' is-current' : ''),
          onClick: () => { onPick(row.id); closeAndReturnFocus() },
          'aria-current': row.id === selectedId ? 'true' : undefined,
        },
        h('span', { className: 'dsh-nt-session-state' + (row.running ? ' is-running' : ''), 'aria-hidden': 'true' }),
        h('span', { className: 'dsh-nt-session-main' },
          h('span', { className: 'dsh-nt-session-title' }, row.displayTitle || row.id),
          h('span', { className: 'dsh-nt-session-meta' },
            row.running ? t('session.running') : timeAgo(row.updatedAt, locale, t),
            row.cwd ? h('span', { className: 'dsh-nt-session-cwd' }, clip(row.cwd, 42)) : null
          )
        ),
        row.id === selectedId ? h('span', { className: 'dsh-nt-session-check', 'aria-hidden': 'true' }, h(IconCheck)) : null
        ))
        : h('div', { className: 'dsh-nt-session-empty' },
          h('strong', null, t('session.empty')),
          h('span', null, t('session.emptyHint'))
        )

      const projectBody = workspaceRows.length
        ? workspaceRows.map(workspace => h('button', {
          key: workspace.id,
          type: 'button',
          className: 'dsh-nt-session-row' + (workspace.id === selectedWorkspaceId ? ' is-current' : ''),
          onClick: () => {
            if (onPickWorkspace) onPickWorkspace(workspace.id)
            closeAndReturnFocus()
          },
          'aria-current': workspace.id === selectedWorkspaceId ? 'true' : undefined,
        },
        h('span', { className: 'dsh-nt-session-state' + (workspace.runningCount ? ' is-running' : ''), 'aria-hidden': 'true' }),
        h('span', { className: 'dsh-nt-session-main' },
          h('span', { className: 'dsh-nt-session-title' }, workspace.title),
          h('span', { className: 'dsh-nt-session-meta' },
            t('project.chatCount', { count: workspace.chatCount }),
            workspace.runningCount ? t('session.runningCount', { count: workspace.runningCount }) : null,
            h('span', { className: 'dsh-nt-session-cwd' }, clip(workspace.path, 46))
          )
        ),
        workspace.id === selectedWorkspaceId ? h('span', { className: 'dsh-nt-session-check', 'aria-hidden': 'true' }, h(IconCheck)) : null
        ))
        : h('div', { className: 'dsh-nt-session-empty' },
          h('strong', null, t('project.noProject')),
          h('span', null, t('project.emptyHint'))
        )

      const body = projectMode ? projectBody : chatBody
      const badgeCount = projectMode ? selectedRunningCount : runningCount
      const pickerTitle = projectMode
        ? (selectedWorkspaceTitle || t('project.noProject'))
        : (selectedTitle || t('panel.title'))
      const pickerLabel = projectMode ? t('project.choose') : t('session.choose')
      const searchLabel = projectMode ? t('project.search') : t('session.search')

      return h('div', {
        className: 'dsh-nt-picker',
        ref: wrapperRef,
        onKeyDown: event => {
          if (open && event.key === 'Escape') {
            event.stopPropagation()
            closeAndReturnFocus()
          }
        },
      },
        h('button', {
          ref: buttonRef,
          type: 'button',
          className: 'dsh-nt-picker-button',
          'aria-haspopup': 'dialog',
          'aria-expanded': open ? 'true' : 'false',
          onClick: () => setOpen(value => !value),
          title: pickerLabel,
        },
        h('span', { className: 'dsh-nt-picker-text' },
          h('span', { className: 'dsh-nt-picker-eyebrow' },
            pickerLabel,
            badgeCount > 0 ? h('span', { className: 'dsh-nt-picker-running' },
              h('span', { className: 'dsh-nt-picker-running-dot', 'aria-hidden': 'true' }),
              t('session.runningCount', { count: badgeCount })
            ) : null
          ),
          h('span', { className: 'dsh-nt-picker-title' }, pickerTitle)
        ),
        h('span', { className: 'dsh-nt-picker-chevron', 'aria-hidden': 'true' }, h(IconChevron))
        ),
        open ? h('div', { className: 'dsh-nt-picker-pop', role: 'dialog', 'aria-label': pickerLabel },
          h('div', { className: 'dsh-nt-picker-search' },
            h(IconSearch),
            h('input', {
              ref: inputRef,
              value: query,
              placeholder: searchLabel,
              'aria-label': searchLabel,
              onChange: event => setQuery(event.target.value),
              onKeyDown: event => {
                if (event.key === 'Escape') {
                  event.stopPropagation()
                  closeAndReturnFocus()
                }
              },
            })
          ),
          h('div', { className: 'dsh-nt-session-list', 'data-nt-scroll': 'true' }, body),
          h('div', { className: 'dsh-nt-picker-foot' },
            h('span', null, projectMode
              ? workspaceRows.length + ' / ' + (workspaces && Array.isArray(workspaces.items) ? workspaces.items.length : 0)
              : rows.length + ' / ' + (list && Array.isArray(list.ids) ? list.ids.length : 0)),
            h('button', {
              type: 'button',
              className: 'dsh-nt-ghost',
              onClick: () => onRefresh(),
              disabled: !!refreshing,
            }, refreshing ? h(IconSpinner) : h(IconRefresh), h('span', null, t('session.refresh')))
          )
        ) : null
      )
    }

    /* ------------------------------------------------------------------ *
     *  Brain stage
     * ------------------------------------------------------------------ */

    function ClassMenu(props) {
      const { classes, active, t, onChange } = props
      const [open, setOpen] = useState(false)
      const ref = useRef(null)
      const buttonRef = useRef(null)

      useEffect(() => {
        if (!open) return undefined
        const onPointerDown = event => {
          if (ref.current && !ref.current.contains(event.target)) setOpen(false)
        }
        const onKeyDown = event => {
          if (event.key === 'Escape') {
            event.stopImmediatePropagation()
            event.preventDefault()
            setOpen(false)
            if (buttonRef.current) buttonRef.current.focus()
          }
        }
        document.addEventListener('pointerdown', onPointerDown, true)
        document.addEventListener('keydown', onKeyDown, true)
        return () => {
          document.removeEventListener('pointerdown', onPointerDown, true)
          document.removeEventListener('keydown', onKeyDown, true)
        }
      }, [open])

      useEffect(() => {
        if (!open || !ref.current) return
        const items = Array.from(ref.current.querySelectorAll('[role="menuitemradio"]'))
        const active = items.find(item => item.getAttribute('aria-checked') === 'true') || items[0]
        if (active) active.focus()
      }, [open])

      function moveFocus(event) {
        if (!ref.current) return
        const items = Array.from(ref.current.querySelectorAll('[role="menuitemradio"]'))
        if (!items.length) return
        const index = items.indexOf(document.activeElement)
        let next = null
        if (event.key === 'ArrowDown') next = (index + 1 + items.length) % items.length
        else if (event.key === 'ArrowUp') next = (index - 1 + items.length) % items.length
        else if (event.key === 'Home') next = 0
        else if (event.key === 'End') next = items.length - 1
        if (next === null) return
        event.preventDefault()
        items[next].focus()
      }

      const rows = Array.isArray(classes) ? classes : []
      const total = rows.reduce((sum, item) => sum + (item.count || 0), 0)
      const activeColor = active ? (CLASSES[active] || CLASSES.context).color : null

      function choose(value) {
        onChange(value)
        setOpen(false)
        if (buttonRef.current) buttonRef.current.focus()
      }

      return h('div', { className: 'dsh-nt-class-wrap', ref },
        h('button', {
          ref: buttonRef,
          type: 'button',
          className: 'dsh-nt-icon-button dsh-nt-class-button' + (active ? ' is-on' : ''),
          title: t('stage.classes'),
          'aria-label': active ? t('class.' + active) : t('stage.allClasses'),
          'aria-haspopup': 'menu',
          'aria-expanded': open ? 'true' : 'false',
          onClick: () => setOpen(value => !value),
        }, h(IconLayers),
        h('span', { className: 'dsh-nt-class-button-label' }, active ? t('class.' + active) : t('stage.classes')),
        active ? h('span', { className: 'dsh-nt-class-dot', style: { background: activeColor } }) : null),
        open ? h('div', {
          className: 'dsh-nt-class-menu',
          role: 'menu',
          'aria-label': t('stage.classes'),
          'data-nt-scroll': 'true',
          onKeyDown: moveFocus,
        },
        h('button', {
          type: 'button',
          role: 'menuitemradio',
          'aria-checked': !active,
          className: 'dsh-nt-class-row' + (!active ? ' is-active' : ''),
          onClick: () => choose(null),
        },
        h('span', { className: 'dsh-nt-class-row-dot is-all' }),
        h('span', { className: 'dsh-nt-class-row-main' },
          h('span', { className: 'dsh-nt-class-row-name' }, t('stage.allClasses')),
          h('span', { className: 'dsh-nt-class-row-hint' }, t('stage.classesHint'))
        ),
        h('span', { className: 'dsh-nt-class-count' }, total),
        !active ? h('span', { className: 'dsh-nt-class-check', 'aria-hidden': 'true' }, h(IconCheck)) : null
        ),
        rows.map(item => h('button', {
          key: item.key,
          type: 'button',
          role: 'menuitemradio',
          'aria-checked': active === item.key,
          className: 'dsh-nt-class-row' + (active === item.key ? ' is-active' : ''),
          style: { '--nt-color': item.color || (CLASSES[item.key] || CLASSES.context).color },
          onClick: () => choose(item.key),
          title: t('class.' + item.key + '.hint'),
        },
        h('span', { className: 'dsh-nt-class-row-dot', style: { background: item.color || (CLASSES[item.key] || CLASSES.context).color } }),
        h('span', { className: 'dsh-nt-class-row-main' },
          h('span', { className: 'dsh-nt-class-row-name' }, t('class.' + item.key)),
          h('span', { className: 'dsh-nt-class-row-hint' }, t('class.' + item.key + '.hint'))
        ),
        h('span', { className: 'dsh-nt-class-count' }, item.count || 0),
        active === item.key ? h('span', { className: 'dsh-nt-class-check', 'aria-hidden': 'true' }, h(IconCheck)) : null
        ))
        ) : null
      )
    }

    const TIMELINE_MARKER_COLORS = {
      prompt: '#ffb86b',
      thought: '#5fe3ff',
      tool: '#ff6bd6',
      answer: '#ffd479',
      document: '#b78bff',
    }

    function Timeline(props) {
      const { runtime, snapshot, t, locale } = props
      const playback = snapshot.playback
      const active = !!playback
      const speed = active ? playback.speed : (snapshot.playbackSpeed || 1)
      const loop = active ? !!playback.loop : !!snapshot.playbackLoop
      const enabled = !!snapshot.sessionId && !snapshot.loading
      const total = active ? playback.total : 0
      const index = active ? playback.index : 0
      const progress = active ? playback.progress : 0
      const currentTime = active ? playback.currentTime : 0
      const endTime = active ? playback.endTime : 0
      const markers = active && Array.isArray(playback.markers) ? playback.markers : []
      const percent = Math.round(progress * 100)

      function toggle() {
        if (!enabled) return
        if (active) runtime.playbackToggle()
        else runtime.playbackStart()
      }

      function onKeyDown(event) {
        const tag = event.target && event.target.tagName
        if (tag === 'INPUT' || tag === 'SELECT' || tag === 'BUTTON') return
        if (event.key === ' ') {
          event.preventDefault()
          toggle()
        } else if (active && event.key === 'ArrowLeft') {
          event.preventDefault()
          runtime.playbackStep(-1)
        } else if (active && event.key === 'ArrowRight') {
          event.preventDefault()
          runtime.playbackStep(1)
        }
      }

      return h('div', {
        className: 'dsh-nt-timeline' + (active ? ' is-active' : '') + (loop ? ' is-loop' : ''),
        role: 'group',
        'aria-label': t('timeline.label'),
        onKeyDown,
      },
      h('div', { className: 'dsh-nt-tl-controls' },
        h('button', {
          type: 'button',
          className: 'dsh-nt-tl-play',
          title: active && playback.playing ? t('timeline.pause') : t('timeline.play'),
          'aria-label': active && playback.playing ? t('timeline.pause') : t('timeline.play'),
          disabled: !enabled,
          onClick: toggle,
        }, active && playback.playing ? h(IconPause) : h(IconPlay)),
        h('button', {
          type: 'button', className: 'dsh-nt-tl-button',
          title: t('timeline.restart'), 'aria-label': t('timeline.restart'),
          disabled: !enabled,
          onClick: () => { if (active) runtime.playbackRestart(); else runtime.playbackStart() },
        }, h(IconRestart)),
        h('button', {
          type: 'button', className: 'dsh-nt-tl-button',
          title: t('timeline.back'), 'aria-label': t('timeline.back'),
          disabled: !active || index <= 0,
          onClick: () => runtime.playbackStep(-1),
        }, h(IconStepBack)),
        h('button', {
          type: 'button', className: 'dsh-nt-tl-button',
          title: t('timeline.forward'), 'aria-label': t('timeline.forward'),
          disabled: !active || index >= total,
          onClick: () => runtime.playbackStep(1),
        }, h(IconStepForward))
      ),
      h('div', { className: 'dsh-nt-tl-main' },
        h('div', { className: 'dsh-nt-tl-meta' },
          h('span', { className: 'dsh-nt-tl-badge' },
            h(IconHistory),
            active ? t('timeline.replaying') : t('timeline.label')),
          h('span', { className: 'dsh-nt-tl-time' },
            active ? clockOf(currentTime, locale) + ' / ' + clockOf(endTime, locale) : t('timeline.ready')),
          active ? h('span', { className: 'dsh-nt-tl-count' }, percent + '%') : null),
        h('div', { className: 'dsh-nt-tl-track' },
          h('input', {
            type: 'range',
            min: 0,
            max: Math.max(1, total),
            step: 1,
            value: active ? index : 0,
            disabled: !active,
            'aria-label': t('timeline.scrub'),
            onChange: event => { if (active) runtime.playbackSeek(Number(event.target.value)) },
          }),
          h('div', { className: 'dsh-nt-tl-markers' },
            markers.map(marker => h('button', {
              key: 'marker:' + marker.index + ':' + marker.kind,
              type: 'button',
              className: 'dsh-nt-tl-marker is-' + marker.kind,
              style: { left: (marker.index / Math.max(1, total) * 100) + '%', background: TIMELINE_MARKER_COLORS[marker.kind] || '#8fb8ff' },
              title: t('timeline.marker') + ' · ' + clockOf(marker.time, locale),
              'aria-label': t('timeline.marker') + ' · ' + clockOf(marker.time, locale),
              onClick: () => runtime.playbackSeek(marker.index),
            }))))
      ),
      h('div', { className: 'dsh-nt-tl-extra' },
        h('label', { className: 'dsh-nt-tl-speed' },
          h('span', { className: 'dsh-nt-sr-only' }, t('timeline.speed')),
          h('select', {
            value: String(speed),
            'aria-label': t('timeline.speed'),
            onChange: event => runtime.playbackSpeed(Number(event.target.value)),
          }, [0.5, 1, 2, 4].map(value => h('option', { key: value, value: String(value) }, value + '×')))),
        h('button', {
          type: 'button',
          className: 'dsh-nt-tl-button' + (loop ? ' is-on' : ''),
          title: t('timeline.loop'),
          'aria-label': t('timeline.loop'),
          'aria-pressed': loop,
          onClick: () => runtime.playbackLoop(),
        }, h(IconLoop)),
        active
          ? h('button', {
            type: 'button', className: 'dsh-nt-tl-live',
            title: t('timeline.live'), 'aria-label': t('timeline.live'),
            onClick: () => runtime.playbackExit(),
          }, h(IconHistory), h('span', null, t('timeline.live')))
          : null
      ))
    }

    function BrainStage(props) {
      const {
        runtime, snapshot, ui, uiStore, t, locale, onSelect, focusRequest, listPending, onRetry,
        onOpenMedia, onOpenFile, loadMedia,
      } = props
      const replaying = !!(snapshot.playback && snapshot.playback.active)
      const canvasRef = useRef(null)
      const stageRef = useRef(null)
      const rendererRef = useRef(null)
      const tRef = useRef(t)
      tRef.current = t
      const selectRef = useRef(onSelect)
      selectRef.current = onSelect
      const openSessionRef = useRef(props.onOpenSession)
      openSessionRef.current = props.onOpenSession
      const [toasts, setToasts] = useState([])
      const [mediaToast, setMediaToast] = useState(null)
      const [fileToast, setFileToast] = useState(null)
      const [hover, setHover] = useState(null)
      const [sectorHover, setSectorHover] = useState(null)
      const [infoOpen, setInfoOpen] = useState(false)
      const sessionRef = useRef(null)
      const eventRef = useRef(0)
      const seenEventsRef = useRef(new Set())
      const mediaSessionRef = useRef(null)
      const mediaRef = useRef(0)
      const seenMediaRef = useRef(new Set())
      const fileSessionRef = useRef(null)
      const fileRef = useRef(0)
      const seenFilesRef = useRef(new Set())
      const projectMode = snapshot.mode === 'project'
      const project = snapshot.project

      useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return undefined
        const renderer = createRenderer(canvas, {
          labelOf: node => {
            const translator = tRef.current
            if (node.family === 'skill') {
              return translator('calls.skillTag') + ' · ' + (node.skillName || node.title || node.toolName || '')
            }
            if (node.family === 'mcp') {
              return translator('calls.mcpTag') + ' · ' + [node.mcpServer, node.mcpTool || node.title].filter(Boolean).join(' / ')
            }
            if (node.titleKey) return translator(node.titleKey)
            return node.title || node.toolName || ''
          },
          onSelect: id => selectRef.current(id),
          onHover: info => setHover(info),
          onActivate: id => {
            const graph = runtime.currentGraph()
            if (!graph.projectMode) return
            const node = graph.byId.get(id)
            if (node && node.sessionId && openSessionRef.current) openSessionRef.current(node.sessionId)
          },
        })
        rendererRef.current = renderer
        renderer.setGraphSource(() => runtime.currentGraph())
        renderer.setSelection(snapshot.selectedId)
        renderer.setFocusSector(snapshot.mode === 'project' ? null : ui.sector)
        renderer.setClassFilter(snapshot.mode === 'project' ? null : ui.classFilter)
        renderer.setLabels(ui.labels)
        renderer.resize()
        renderer.start()
        let observer = null
        if (typeof ResizeObserver !== 'undefined' && canvas.parentElement) {
          observer = new ResizeObserver(() => renderer.resize())
          observer.observe(canvas.parentElement)
        }
        return () => {
          if (observer) observer.disconnect()
          renderer.destroy()
          rendererRef.current = null
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, [runtime])

      useEffect(() => {
        const renderer = rendererRef.current
        if (!renderer) return
        renderer.setSelection(snapshot.selectedId)
        renderer.setFocusSector(snapshot.mode === 'project' ? null : ui.sector)
        renderer.setClassFilter(snapshot.mode === 'project' ? null : ui.classFilter)
        renderer.setLabels(ui.labels)
      }, [snapshot.revision, snapshot.selectedId, snapshot.mode, ui.sector, ui.classFilter, ui.labels])

      useEffect(() => {
        if (!focusRequest || !rendererRef.current) return
        rendererRef.current.focusNode(focusRequest.id)
      }, [focusRequest])

      useEffect(() => {
        setHover(null)
      }, [snapshot.sessionId, snapshot.mode])

      useEffect(() => {
        if (!replaying) return
        setToasts([])
        setMediaToast(null)
        setFileToast(null)
        setHover(null)
      }, [replaying])

      useEffect(() => {
        if (!infoOpen) return undefined
        const onKeyDown = event => {
          if (event.key !== 'Escape') return
          event.stopImmediatePropagation()
          event.preventDefault()
          setInfoOpen(false)
        }
        document.addEventListener('keydown', onKeyDown, true)
        return () => document.removeEventListener('keydown', onKeyDown, true)
      }, [infoOpen])

      useEffect(() => {
        if (snapshot.mode === 'project' || !snapshot.sessionId) {
          sessionRef.current = null
          seenEventsRef.current.clear()
          setToasts(previous => previous.length ? [] : previous)
          return
        }
        if (sessionRef.current !== snapshot.sessionId) {
          sessionRef.current = snapshot.sessionId
          eventRef.current = snapshot.loadedAt || Date.now()
          seenEventsRef.current.clear()
          setToasts([])
        }
        const events = snapshot.events || []
        const fresh = events.filter(event =>
          event.time > eventRef.current && (event.live || !seenEventsRef.current.has(event.id)))
        let newest = eventRef.current
        for (const event of events) {
          newest = Math.max(newest, event.time)
          if (event.time <= eventRef.current) seenEventsRef.current.add(event.id)
        }
        eventRef.current = newest
        if (!fresh.length) return
        for (const event of fresh) seenEventsRef.current.add(event.id)
        if (seenEventsRef.current.size > 600) {
          seenEventsRef.current = new Set(Array.from(seenEventsRef.current).slice(-300))
        }
        const picked = pickToastEvents(fresh)
        const now = Date.now()
        const added = picked
          .filter(Boolean)
          .map(event => Object.assign({}, event, { expiresAt: now + 6000 }))
        if (!added.length) return
        setToasts(previous => {
          const byId = new Map(previous.map(toast => [toast.id, toast]))
          for (const toast of added) byId.set(toast.id, toast)
          return Array.from(byId.values()).slice(snapshot.live ? -2 : -3)
        })
      }, [snapshot.revision, snapshot.mode, snapshot.sessionId, snapshot.loadedAt, snapshot.events])

      useEffect(() => {
        if (snapshot.mode === 'project' || !snapshot.sessionId) {
          mediaSessionRef.current = null
          seenMediaRef.current.clear()
          setMediaToast(null)
          return
        }
        if (mediaSessionRef.current !== snapshot.sessionId) {
          mediaSessionRef.current = snapshot.sessionId
          mediaRef.current = snapshot.loadedAt || Date.now()
          seenMediaRef.current.clear()
          setMediaToast(null)
        }
        const events = snapshot.mediaEvents || []
        const fresh = events.filter(item =>
          item.time > mediaRef.current && !seenMediaRef.current.has(item.id))
        let newest = mediaRef.current
        for (const item of events) {
          newest = Math.max(newest, item.time)
          if (item.time <= mediaRef.current) seenMediaRef.current.add(item.id)
        }
        mediaRef.current = newest
        if (!fresh.length) return
        for (const item of fresh) seenMediaRef.current.add(item.id)
        if (seenMediaRef.current.size > 300) {
          seenMediaRef.current = new Set(Array.from(seenMediaRef.current).slice(-150))
        }
        const media = snapshot.media || []
        const ordered = fresh.slice().sort((a, b) => a.time - b.time)
        const latest = ordered[ordered.length - 1]
        const item = media.find(entry => entry.id === latest.mediaId)
        if (!item) return
        setMediaToast({ media: item, extra: ordered.length - 1, expiresAt: Date.now() + 9000 })
      }, [snapshot.revision, snapshot.mode, snapshot.sessionId, snapshot.loadedAt, snapshot.mediaEvents, snapshot.media])

      useEffect(() => {
        if (!mediaToast) return undefined
        const timer = setTimeout(() => setMediaToast(null), Math.max(0, mediaToast.expiresAt - Date.now()))
        return () => clearTimeout(timer)
      }, [mediaToast])

      useEffect(() => {
        if (snapshot.mode === 'project' || !snapshot.sessionId) {
          fileSessionRef.current = null
          seenFilesRef.current.clear()
          setFileToast(null)
          return
        }
        if (fileSessionRef.current !== snapshot.sessionId) {
          fileSessionRef.current = snapshot.sessionId
          fileRef.current = snapshot.loadedAt || Date.now()
          seenFilesRef.current.clear()
          setFileToast(null)
        }
        const events = snapshot.fileEvents || []
        const fresh = events.filter(item => item.time > fileRef.current && !seenFilesRef.current.has(item.id))
        let newest = fileRef.current
        for (const item of events) {
          newest = Math.max(newest, item.time)
          if (item.time <= fileRef.current) seenFilesRef.current.add(item.id)
        }
        fileRef.current = newest
        if (!fresh.length) return
        for (const item of fresh) seenFilesRef.current.add(item.id)
        if (seenFilesRef.current.size > 300) {
          seenFilesRef.current = new Set(Array.from(seenFilesRef.current).slice(-150))
        }
        const files = snapshot.files || []
        const ordered = fresh.slice().sort((a, b) => a.time - b.time)
        const latest = ordered[ordered.length - 1]
        const file = files.find(entry => entry.id === latest.fileId)
        if (!file) return
        setFileToast({ file, extra: ordered.length - 1, expiresAt: Date.now() + 9000 })
      }, [snapshot.revision, snapshot.mode, snapshot.sessionId, snapshot.loadedAt, snapshot.fileEvents, snapshot.files])

      useEffect(() => {
        if (!fileToast) return undefined
        const timer = setTimeout(() => setFileToast(null), Math.max(0, fileToast.expiresAt - Date.now()))
        return () => clearTimeout(timer)
      }, [fileToast])

      useEffect(() => {
        const timer = setInterval(() => {
          setToasts(previous => {
            const now = Date.now()
            const next = previous.filter(toast => toast.expiresAt > now)
            return next.length === previous.length ? previous : next
          })
        }, 800)
        return () => clearInterval(timer)
      }, [])

      const sectors = snapshot.sectors || []
      const classes = snapshot.classes || []
      const activeClass = projectMode ? null : ui.classFilter
      const empty = projectMode ? !project : !snapshot.sessionId
      const quiet = projectMode
        ? !!(project && project.total === 0 && !snapshot.error)
        : !replaying && !!(snapshot.sessionId && !snapshot.loading && !snapshot.error && snapshot.eventCount === 0)
      const latestSignal = snapshot.signals && snapshot.signals.length ? snapshot.signals[0] : null
      const announcement = latestSignal
        ? latestSignal.toolName + ' · ' + sectorLabel(latestSignal.sector, t, true)
        : ''
      const selectionAnnouncement = snapshot.selected
        ? kindLabel(snapshot.selected.kind, t) + ' · ' + nodeTitle(snapshot.selected, t)
          + ' · ' + sectorLabel(snapshot.selected.sector, t, true)
        : ''

      function selectFromToast(toast) {
        onSelect(toast.id)
        setToasts(previous => previous.filter(item => item.id !== toast.id))
      }

      let hoverNode = null
      if (hover) {
        const currentGraph = runtime.currentGraph()
        hoverNode = currentGraph.byId.get(hover.id) || null
      }
      let tipStyle = null
      let tipContent = null
      if (hover && hoverNode) {
        const stageWidth = stageRef.current ? stageRef.current.clientWidth : 0
        const stageHeight = stageRef.current ? stageRef.current.clientHeight : 0
        tipStyle = {
          left: clamp(hover.x + 16, 8, Math.max(8, stageWidth - 312)),
          top: clamp(hover.y + 16, 8, Math.max(8, stageHeight - 172)),
        }
        const snippet = clip(
          hoverNode.text || hoverNode.summary || (hoverNode.argsRaw ? prettyJson(hoverNode.argsRaw) : '') || hoverNode.result || '',
          260,
        )
        const hoverFamily = hoverNode.family === 'skill' || hoverNode.family === 'mcp' ? hoverNode.family : null
        tipContent = h('div', { className: 'dsh-nt-tip' + (hoverFamily ? ' is-call is-' + hoverFamily : ''), style: tipStyle, role: 'tooltip' },
          h('div', { className: 'dsh-nt-tip-head' },
            h('span', { className: 'dsh-nt-tip-dot', style: { background: hoverFamily ? callFamilyColor(hoverFamily) : (hoverNode.color || (SECTORS[hoverNode.sector] || SECTORS.parietal).color) } }),
            h('span', { className: 'dsh-nt-tip-kind' }, hoverFamily
              ? t(hoverFamily === 'skill' ? 'calls.skill' : 'calls.mcp')
              : kindLabel(hoverNode.kind, t)),
            h('span', { className: 'dsh-nt-tip-sector' }, hoverNode.kind === 'project'
              ? t('project.legendHub')
              : hoverNode.kind === 'session'
                ? t('project.legendChat')
                : hoverNode.kind === 'subagent-session'
                  ? t('project.legendLink')
                  : sectorLabel(hoverNode.sector, t, true)),
            hoverFamily === 'mcp' && hoverNode.mcpServer
              ? h('span', { className: 'dsh-nt-tip-family' }, hoverNode.mcpServer)
              : null,
            h('span', { className: 'dsh-nt-tip-status ' + hoverNode.status }, statusLabel(hoverNode.status, t))
          ),
          h('div', { className: 'dsh-nt-tip-title' }, nodeTitle(hoverNode, t)),
          snippet ? h('div', { className: 'dsh-nt-tip-text' }, snippet) : null
        )
      }

      return h('section', { className: 'dsh-nt-stage', ref: stageRef },
        h('canvas', {
          ref: canvasRef,
          className: 'dsh-nt-canvas',
          tabIndex: 0,
          'aria-label': t('stage.canvasLabel'),
          'aria-describedby': 'dsh-nt-canvas-help',
        }),
        h('span', { id: 'dsh-nt-canvas-help', className: 'dsh-nt-sr-only' }, t('stage.keyboard')),
        h('div', { className: 'dsh-nt-sr-only', role: 'status', 'aria-live': 'polite' }, announcement),
        h('div', { className: 'dsh-nt-sr-only', role: 'status', 'aria-live': 'polite' }, selectionAnnouncement),

        h('div', { className: 'dsh-nt-hud-top' },
          h('div', { className: 'dsh-nt-hud-title' },
            h('span', { className: 'dsh-nt-hud-kicker' }, projectMode ? t('stage.kickerProject') : t('stage.kicker')),
            h('span', { className: 'dsh-nt-hud-counts' }, t('stage.nodes', {
              nodes: snapshot.nodeCount || 0,
              edges: snapshot.edgeCount || 0,
            })),
            !projectMode && snapshot.callCounts && snapshot.callCounts.total
              ? h('span', { className: 'dsh-nt-hud-calls' },
                ['skill', 'mcp'].map(key => snapshot.callCounts[key]
                  ? h('button', {
                    key,
                    type: 'button',
                    className: 'dsh-nt-hud-call',
                    style: { '--nt-color': callFamilyColor(key) },
                    title: t(key === 'skill' ? 'calls.skillHint' : 'calls.mcpHint'),
                    onClick: () => uiStore.patch({ tab: 'calls' }),
                  },
                  h(CallFamilyIcon, { family: key }),
                  h('span', null, t(key === 'skill' ? 'calls.skill' : 'calls.mcp')),
                  h('span', { className: 'dsh-nt-hud-call-count' }, snapshot.callCounts[key]))
                  : null))
              : null,
            activeClass
              ? h('span', { className: 'dsh-nt-class-active', style: { '--nt-color': (CLASSES[activeClass] || CLASSES.context).color } },
                h('span', { className: 'dsh-nt-class-dot', style: { background: (CLASSES[activeClass] || CLASSES.context).color } }),
                t('class.' + activeClass))
              : null,
            snapshot.hasMore
              ? h('span', { className: 'dsh-nt-hud-hint' }, projectMode
                ? t('project.partial', {
                  shown: project ? project.shown : 0,
                  total: project ? project.total : 0,
                })
                : t('stage.partial'))
              : null,
            !projectMode && snapshot.capped && snapshot.dropped
              ? h('span', { className: 'dsh-nt-hud-cap' }, t('stage.capped', { dropped: snapshot.dropped }))
              : null
          ),
          h('div', { className: 'dsh-nt-hud-actions' },
            projectMode ? null : h(ClassMenu, {
              classes: classes.length ? classes : CLASS_KEYS.map(key => ({ key, count: 0, color: CLASSES[key].color })),
              active: ui.classFilter,
              t,
              onChange: value => uiStore.patch({ classFilter: value }),
            }),
            h('button', {
              type: 'button',
              className: 'dsh-nt-icon-button' + (ui.labels ? ' is-on' : ''),
              title: t('stage.labels'),
              'aria-label': t('stage.labels'),
              'aria-pressed': ui.labels,
              onClick: () => uiStore.patch({ labels: !ui.labels }),
            }, h(IconLabel)),
            h('button', {
              type: 'button', className: 'dsh-nt-icon-button', title: t('stage.zoomIn'),
              'aria-label': t('stage.zoomIn'),
              onClick: () => rendererRef.current && rendererRef.current.zoomBy(1.18),
            }, h(IconPlus)),
            h('button', {
              type: 'button', className: 'dsh-nt-icon-button', title: t('stage.zoomOut'),
              'aria-label': t('stage.zoomOut'),
              onClick: () => rendererRef.current && rendererRef.current.zoomBy(1 / 1.18),
            }, h(IconMinus)),
            h('button', {
              type: 'button', className: 'dsh-nt-icon-button', title: t('stage.reset'),
              'aria-label': t('stage.reset'),
              onClick: () => rendererRef.current && rendererRef.current.resetView(),
            }, h(IconOrbit))
          )
        ),

        (!replaying && (snapshot.live || mediaToast || fileToast || toasts.length)) ? h('div', { className: 'dsh-nt-toasts' },
          snapshot.live ? (() => {
            const live = snapshot.live
            const liveColor = live.color || (SECTORS[live.sector] || SECTORS.parietal).color
            const full = String(live.text || '')
            const tail = full.length > 560 ? '…' + full.slice(-560) : full
            return h('button', {
              type: 'button',
              className: 'dsh-nt-live-thought',
              style: { '--nt-color': liveColor },
              onClick: () => onSelect(live.id),
              title: t('node.inspect'),
            },
            h('span', { className: 'dsh-nt-live-thought-head' },
              h('span', { className: 'dsh-nt-live-thought-icon', 'aria-hidden': 'true' }, h(IconThought)),
              h('span', { className: 'dsh-nt-live-thought-kind' }, kindLabel(live.kind, t)),
              h('span', { className: 'dsh-nt-live-thought-sector' }, sectorLabel(live.sector, t, true)),
              h('span', { className: 'dsh-nt-live-thought-badge' },
                h('span', { className: 'dsh-nt-live-dot', 'aria-hidden': 'true' }),
                t('flow.live'))
            ),
            h('span', { className: 'dsh-nt-live-thought-text' }, tail,
              h('span', { className: 'dsh-nt-caret', 'aria-hidden': 'true' }))
            )
          })() : null,
          mediaToast ? (() => {
            const item = mediaToast.media
            const color = item.color || (SECTORS[item.sector] || SECTORS.parietal).color
            const summary = [
              item.kind === 'video' ? t('media.video') : t('media.image'),
              item.width && item.height ? item.width + '×' + item.height : '',
              item.bytes ? formatBytes(item.bytes) : '',
              mediaToast.extra > 0 ? t('media.moreArrived', { count: mediaToast.extra }) : '',
            ].filter(Boolean).join(' · ')
            return h('button', {
              type: 'button',
              className: 'dsh-nt-toast is-media',
              style: { '--nt-color': color },
              onClick: () => {
                setMediaToast(null)
                if (onOpenMedia) onOpenMedia(item)
              },
              title: t('media.open'),
            },
            h('span', { className: 'dsh-nt-media-toast-thumb' },
              h(MediaThumb, {
                media: item,
                sessionId: snapshot.sessionId,
                loadMedia,
                alt: item.name,
              })),
            h('span', { className: 'dsh-nt-toast-body' },
              h('span', { className: 'dsh-nt-toast-name' }, item.name || t('media.untitled')),
              h('span', { className: 'dsh-nt-toast-summary' }, summary)),
            h('span', { className: 'dsh-nt-toast-sector' },
              item.toolName || (item.source === 'prompt' ? t('media.fromPrompt') : t('media.source')))
            )
          })() : null,
          fileToast ? (() => {
            const item = fileToast.file
            return h('button', {
              type: 'button',
              className: 'dsh-nt-toast is-file',
              style: { '--nt-color': '#8fb8ff' },
              onClick: () => {
                setFileToast(null)
                if (onOpenFile) onOpenFile(item)
              },
              title: t('file.open'),
            },
            h('span', { className: 'dsh-nt-file-toast-icon' }, h(IconDoc)),
            h('span', { className: 'dsh-nt-toast-body' },
              h('span', { className: 'dsh-nt-toast-name' }, item.name),
              h('span', { className: 'dsh-nt-toast-summary' },
                [item.ext.toUpperCase(), item.toolName || t('media.source'), fileToast.extra > 0 ? t('file.moreArrived', { count: fileToast.extra }) : ''].filter(Boolean).join(' · '))),
            h('span', { className: 'dsh-nt-toast-sector' }, item.group ? t('file.' + item.group) : t('file.text')))
          })() : null,
          toasts.map(toast => {
            const family = toast.family === 'skill' || toast.family === 'mcp' ? toast.family : null
            const color = family ? callFamilyColor(family) : (toast.color || (SECTORS[toast.sector] || SECTORS.parietal).color)
            const thought = toast.kind === 'reasoning' || toast.kind === 'answer' || toast.kind === 'attempt'
            const name = family === 'skill'
              ? (toast.skillName || toast.title || toast.toolName)
              : family === 'mcp'
                ? [toast.mcpServer || 'MCP', toast.mcpTool || toast.title].filter(Boolean).join(' / ')
                : toast.kind === 'prompt' || thought
                  ? kindLabel(toast.kind, t)
                  : (toast.titleKey ? t(toast.titleKey) : toast.title)
                    || toast.toolName
                    || kindLabel(toast.kind, t)
            return h('button', {
              key: toast.id + ':' + toast.expiresAt,
              type: 'button',
              className: 'dsh-nt-toast' + (family ? ' is-call is-' + family : ''),
              style: { '--nt-color': color },
              onClick: () => selectFromToast(toast),
              title: (family ? t(family === 'skill' ? 'calls.skill' : 'calls.mcp') : kindLabel(toast.kind, t))
                + ' · ' + (family ? name : sectorLabel(toast.sector, t, true)),
            },
            h('span', { className: 'dsh-nt-toast-icon' }, family ? h(CallFamilyIcon, { family }) : h(EventIcon, { kind: toast.kind })),
            h('span', { className: 'dsh-nt-toast-body' },
              h('span', { className: 'dsh-nt-toast-name' }, name),
              toast.summary ? h('span', { className: 'dsh-nt-toast-summary' }, toast.summary) : null
            ),
            h('span', { className: 'dsh-nt-toast-sector' }, family
              ? t(family === 'skill' ? 'calls.skill' : 'calls.mcp')
              : sectorLabel(toast.sector, t, true))
            )
          })
        ) : null,

        !projectMode ? h(Timeline, { runtime, snapshot, t, locale }) : null,

        projectMode
          ? h('div', { className: 'dsh-nt-legend dsh-nt-project-legend' },
            h('span', { className: 'dsh-nt-legend-chip is-static', style: { '--nt-color': '#ffd479' } },
              h('span', { className: 'dsh-nt-legend-dot' }), h('span', { className: 'dsh-nt-legend-label' }, t('project.legendHub'))),
            h('span', { className: 'dsh-nt-legend-chip is-static', style: { '--nt-color': '#7fb8ff' } },
              h('span', { className: 'dsh-nt-legend-dot' }), h('span', { className: 'dsh-nt-legend-label' }, t('project.legendChat'))),
            h('span', { className: 'dsh-nt-legend-chip is-static', style: { '--nt-color': '#b78bff' } },
              h('span', { className: 'dsh-nt-legend-dot' }), h('span', { className: 'dsh-nt-legend-label' }, t('project.legendLink'))),
            h('span', { className: 'dsh-nt-legend-chip is-static', style: { '--nt-color': '#5f9dff' } },
              h('span', { className: 'dsh-nt-legend-dot' }), h('span', { className: 'dsh-nt-legend-label' }, t('project.legendTime')))
          )
          : h('div', { className: 'dsh-nt-legend', 'aria-label': t('stage.legend') },
            sectors.map(sector => {
              const active = ui.sector === sector.key
              return h('button', {
                key: sector.key,
                type: 'button',
                className: 'dsh-nt-legend-chip' + (active ? ' is-active' : ''),
                style: { '--nt-color': (SECTORS[sector.key] || SECTORS.parietal).color },
                'aria-pressed': active,
                'aria-label': sectorLabel(sector.key, t, false) + '. ' + t('sector.' + sector.key + '.hint'),
                onMouseEnter: () => setSectorHover(sector.key),
                onMouseLeave: () => setSectorHover(current => current === sector.key ? null : current),
                onFocus: () => setSectorHover(sector.key),
                onBlur: () => setSectorHover(current => current === sector.key ? null : current),
                onClick: () => uiStore.patch({ sector: active ? null : sector.key }),
              },
              h('span', { className: 'dsh-nt-legend-dot', 'aria-hidden': 'true' }),
              h('span', { className: 'dsh-nt-legend-label' }, sectorLabel(sector.key, t, true)),
              h('span', { className: 'dsh-nt-legend-count' }, sector.count)
              )
            }),
            h('button', {
              type: 'button',
              className: 'dsh-nt-legend-chip dsh-nt-legend-info' + (infoOpen ? ' is-active' : ''),
              title: t('stage.info'),
              'aria-label': t('stage.info'),
              'aria-pressed': infoOpen,
              onClick: () => setInfoOpen(value => !value),
            }, h(IconHelp))
          ),

        sectorHover && !projectMode && !infoOpen
          ? h('div', { className: 'dsh-nt-sector-note', style: { '--nt-color': (SECTORS[sectorHover] || SECTORS.parietal).color } },
            h('div', { className: 'dsh-nt-sector-note-head' },
              h('span', { className: 'dsh-nt-legend-dot', 'aria-hidden': 'true' }),
              h('strong', null, sectorLabel(sectorHover, t, false)),
              h('span', { className: 'dsh-nt-sector-note-count' }, (sectors.find(item => item.key === sectorHover) || {}).count || 0)
            ),
            h('p', null, t('sector.' + sectorHover + '.hint'))
          )
          : null,

        infoOpen && !projectMode
          ? h('div', { className: 'dsh-nt-sector-guide', 'data-nt-scroll': 'true' },
            h('div', { className: 'dsh-nt-sector-guide-head' },
              h('strong', null, t('stage.info')),
              h('button', {
                type: 'button',
                className: 'dsh-nt-icon-button',
                title: t('stage.infoClose'),
                'aria-label': t('stage.infoClose'),
                onClick: () => setInfoOpen(false),
              }, h(IconClose))
            ),
            sectors.map(sector => h('button', {
              key: sector.key,
              type: 'button',
              className: 'dsh-nt-sector-guide-row' + (ui.sector === sector.key ? ' is-active' : ''),
              style: { '--nt-color': (SECTORS[sector.key] || SECTORS.parietal).color },
              onClick: () => uiStore.patch({ sector: ui.sector === sector.key ? null : sector.key }),
            },
            h('span', { className: 'dsh-nt-legend-dot', 'aria-hidden': 'true' }),
            h('span', { className: 'dsh-nt-sector-guide-text' },
              h('span', { className: 'dsh-nt-sector-guide-name' }, sectorLabel(sector.key, t, false)),
              h('span', { className: 'dsh-nt-sector-guide-hint' }, t('sector.' + sector.key + '.hint'))
            ),
            h('span', { className: 'dsh-nt-legend-count' }, sector.count)
            ))
          )
          : null,

        tipContent,

        h('div', { className: 'dsh-nt-stage-hint' }, projectMode ? t('stage.hintProject') : t('stage.hint')),

        snapshot.loading || (!projectMode && empty && listPending) ? h('div', { className: 'dsh-nt-stage-overlay' },
          h(IconSpinner),
          h('span', null, t('loading'))
        ) : null,

        snapshot.error ? h('div', { className: 'dsh-nt-stage-overlay is-error' },
          h(IconAlert),
          h('strong', null, t('error', { message: snapshot.error })),
          h('button', {
            type: 'button', className: 'dsh-nt-ghost',
            onClick: () => { if (onRetry) onRetry(); else runtime.reload() },
          }, t('retry'))
        ) : null,

        !projectMode && empty && !listPending ? h('div', { className: 'dsh-nt-stage-overlay is-empty' },
          h('strong', null, t('session.empty')),
          h('span', null, t('session.emptyHint'))
        ) : null,

        projectMode && empty ? h('div', { className: 'dsh-nt-stage-overlay is-empty' },
          h('strong', null, t('project.empty')),
          h('span', null, t('project.emptyHint'))
        ) : null,

        quiet ? h('div', { className: 'dsh-nt-stage-overlay is-empty' },
          h('strong', null, projectMode ? t('project.empty') : t('stage.empty')),
          h('span', null, projectMode ? t('project.emptyHint') : t('stage.emptyHint'))
        ) : null
      )
    }

    /* ------------------------------------------------------------------ *
     *  Side panels
     * ------------------------------------------------------------------ */

    function formatBytes(value) {
      const n = Number(value) || 0
      if (!n) return ''
      if (n < 1024) return n + ' B'
      if (n < 1024 * 1024) return (n / 1024).toFixed(1).replace(/\.0$/, '') + ' KB'
      return (n / (1024 * 1024)).toFixed(1).replace(/\.0$/, '') + ' MB'
    }

    function mediaFileUrl(path) {
      if (!path) return ''
      if (typeof window === 'undefined' || !window.location) return ''
      return '/api/file?path=' + encodeURIComponent(path)
    }

    /** Resolve a durable attachment through ui-conversation, or a local path through /api/file. */
    function useMediaUrl(media, sessionId, loadMedia) {
      const local = media && media.path ? mediaFileUrl(media.path) : ''
      const [state, setState] = useState({ url: local, error: null, loading: !!media && !local })
      useEffect(() => {
        if (!media) {
          setState({ url: '', error: null, loading: false })
          return undefined
        }
        if (media.path) {
          setState({ url: mediaFileUrl(media.path), error: null, loading: false })
          return undefined
        }
        if (!media.attachment || !sessionId || typeof loadMedia !== 'function') {
          setState({ url: '', error: 'unavailable', loading: false })
          return undefined
        }
        let cancelled = false
        let pending
        setState({ url: '', error: null, loading: true })
        try {
          pending = loadMedia(sessionId, media.attachment)
        } catch (error) {
          setState({ url: '', error: error || 'failed', loading: false })
          return undefined
        }
        Promise.resolve(pending)
          .then(url => { if (!cancelled) setState({ url: url || '', error: null, loading: false }) })
          .catch(error => { if (!cancelled) setState({ url: '', error: error || 'failed', loading: false }) })
        return () => { cancelled = true }
      }, [media && media.id, sessionId, loadMedia])
      return state
    }

    function MediaThumb(props) {
      const { media, sessionId, loadMedia, className, alt, controls } = props
      const state = useMediaUrl(media, sessionId, loadMedia)
      if (!media) return null
      if (state.url && media.kind === 'video' && controls) {
        return h('video', {
          className: className,
          src: state.url,
          controls: true,
          preload: 'metadata',
        })
      }
      if (state.url && media.kind !== 'video') {
        return h('img', {
          className: className,
          src: state.url,
          alt: alt || media.name || '',
          loading: 'lazy',
          decoding: 'async',
        })
      }
      return h('span', { className: (className || '') + ' dsh-nt-media-placeholder' + (state.error ? ' is-error' : '') },
        state.loading ? h(IconSpinner) : state.error ? h(IconAlert) : media.kind === 'video' ? h(IconVideo) : h(IconImage))
    }

    function MediaViewer(props) {
      const { media, sessionId, loadMedia, t, locale, onClose, onSelectNode } = props
      const viewer = useMediaUrl(media, sessionId, loadMedia)

      useEffect(() => {
        const onKeyDown = event => {
          if (event.key !== 'Escape') return
          event.stopImmediatePropagation()
          event.preventDefault()
          onClose()
        }
        document.addEventListener('keydown', onKeyDown, true)
        return () => document.removeEventListener('keydown', onKeyDown, true)
      }, [onClose])

      if (!media) return null

      function sourceLabel(item) {
        if (item.toolName) return item.toolName
        if (item.source === 'prompt') return t('media.fromPrompt')
        if (item.source === 'deliverable') return t('media.deliverable')
        return t('media.source')
      }

      const viewerMeta = [
        { key: 'media.kind', value: media.kind === 'video' ? t('media.video') : t('media.image') },
      ]
      if (media.width && media.height) viewerMeta.push({ key: 'media.dimensions', value: media.width + ' × ' + media.height })
      if (media.bytes) viewerMeta.push({ key: 'media.size', value: formatBytes(media.bytes) })
      viewerMeta.push({ key: 'media.source', value: sourceLabel(media) })
      viewerMeta.push({ key: 'media.time', value: timeAgo(media.time, locale, t) })

      return h('div', { className: 'dsh-nt-media-viewer', role: 'dialog', 'aria-modal': 'true', 'aria-label': media.name || t('media.preview') },
        h('button', {
          type: 'button',
          className: 'dsh-nt-media-viewer-backdrop',
          onClick: onClose,
          'aria-label': t('media.close'),
        }),
        h('div', { className: 'dsh-nt-media-viewer-card', 'data-nt-scroll': 'true' },
          h('div', { className: 'dsh-nt-media-viewer-head' },
            h('span', { className: 'dsh-nt-media-viewer-title' }, media.name || t('media.untitled')),
            h('button', {
              type: 'button',
              className: 'dsh-nt-icon-button',
              title: t('media.close'),
              'aria-label': t('media.close'),
              onClick: onClose,
            }, h(IconClose))),
          h('div', { className: 'dsh-nt-media-viewer-stage', 'data-nt-scroll': 'true' },
            viewer.loading
              ? h('span', { className: 'dsh-nt-media-viewer-status' }, h(IconSpinner), t('media.loading'))
              : viewer.error
                ? h('span', { className: 'dsh-nt-media-viewer-status is-error' }, h(IconAlert), t('media.loadFailed'))
                : media.kind === 'video'
                  ? h('video', { src: viewer.url, controls: true, preload: 'metadata', className: 'dsh-nt-media-viewer-video' })
                  : h('img', { src: viewer.url, alt: media.name || '', className: 'dsh-nt-media-viewer-image' })),
          h('div', { className: 'dsh-nt-media-viewer-meta' },
            viewerMeta.map(row => h('div', { className: 'dsh-nt-media-meta-row', key: row.key },
              h('span', { className: 'dsh-nt-media-meta-key' }, t(row.key)),
              h('span', { className: 'dsh-nt-media-meta-value' }, row.value)))),
          media.path ? h('div', { className: 'dsh-nt-media-viewer-path', title: media.path }, media.path) : null,
          h('div', { className: 'dsh-nt-media-viewer-actions' },
            media.nodeId
              ? h('button', {
                type: 'button',
                className: 'dsh-nt-primary',
                onClick: () => { onClose(); onSelectNode(media.nodeId) },
              }, h(IconArrowRight), h('span', null, t('media.openNode')))
              : null,
            viewer.url
              ? h('a', {
                className: 'dsh-nt-ghost',
                href: viewer.url,
                download: media.name || 'media',
              }, h(IconDownload), h('span', null, t('media.download')))
              : null)))
    }

    function MediaPanel(props) {
      const {
        media, sessionId, loadMedia, t, locale, selectedId,
        onOpen, onClose, onSelectNode, dropped, hasMore, loadingOlder, onLoadOlder,
      } = props
      const items = Array.isArray(media) ? media : []
      const [filter, setFilter] = useState('all')
      const [limit, setLimit] = useState(18)
      const imageCount = items.filter(item => item.kind === 'image').length
      const videoCount = items.filter(item => item.kind === 'video').length
      const filtered = filter === 'all' ? items : items.filter(item => item.kind === filter)
      const visible = filtered.slice(0, limit)
      const selected = items.find(item => item.id === selectedId) || null

      useEffect(() => {
        setLimit(18)
      }, [filter, sessionId])

      useEffect(() => {
        if (selectedId && !selected) onClose()
      }, [selectedId, selected, onClose])

      if (!items.length) {
        return h('div', { className: 'dsh-nt-media-empty' },
          h('span', { className: 'dsh-nt-media-empty-icon' }, h(IconImage)),
          h('span', { className: 'dsh-nt-media-empty-title' }, t('media.empty')),
          h('span', { className: 'dsh-nt-media-empty-hint' }, t('media.emptyHint')),
          hasMore || loadingOlder
            ? h('button', {
              type: 'button',
              className: 'dsh-nt-media-more is-history',
              disabled: !!loadingOlder,
              onClick: () => { if (onLoadOlder) onLoadOlder() },
            }, loadingOlder ? h(IconSpinner) : h(IconRefresh),
            h('span', null, loadingOlder ? t('media.loadingOlder') : t('media.loadOlder')))
            : null)
      }

      const filterChips = [
        { key: 'all', label: t('media.all'), count: items.length },
        { key: 'image', label: t('media.images'), count: imageCount },
      ]
      if (videoCount) filterChips.push({ key: 'video', label: t('media.videos'), count: videoCount })

      return h('div', { className: 'dsh-nt-media' },
        h('div', { className: 'dsh-nt-media-toolbar' },
          h('div', { className: 'dsh-nt-media-filters' },
            filterChips.map(chip => h('button', {
              key: chip.key,
              type: 'button',
              className: 'dsh-nt-media-filter' + (filter === chip.key ? ' is-active' : ''),
              'aria-pressed': filter === chip.key,
              onClick: () => setFilter(chip.key),
            }, chip.label, h('span', { className: 'dsh-nt-media-filter-count' }, chip.count)))),
          dropped
            ? h('span', { className: 'dsh-nt-media-dropped', title: t('media.droppedHint') }, t('media.dropped', { count: dropped }))
            : null
        ),
        hasMore || loadingOlder
          ? h('button', {
            type: 'button',
            className: 'dsh-nt-media-more is-history',
            disabled: !!loadingOlder,
            onClick: () => { if (onLoadOlder) onLoadOlder() },
          }, loadingOlder ? h(IconSpinner) : h(IconRefresh),
          h('span', null, loadingOlder ? t('media.loadingOlder') : t('media.loadOlder')))
          : null,
        h('div', { className: 'dsh-nt-media-grid' },
          visible.map(item => h('button', {
            key: item.id,
            type: 'button',
            className: 'dsh-nt-media-tile' + (selectedId === item.id ? ' is-active' : ''),
            onClick: () => onOpen(item.id),
            title: item.path || item.name,
          },
          h('span', { className: 'dsh-nt-media-tile-frame' },
            h(MediaThumb, { media: item, sessionId, loadMedia, alt: item.name }),
            item.kind === 'video' ? h('span', { className: 'dsh-nt-media-play', 'aria-hidden': 'true' }, h(IconVideo)) : null),
          h('span', { className: 'dsh-nt-media-tile-body' },
            h('span', { className: 'dsh-nt-media-tile-name' }, item.name || t('media.untitled')),
            h('span', { className: 'dsh-nt-media-tile-meta' },
              item.width && item.height ? item.width + '×' + item.height : item.kind === 'video' ? t('media.video') : t('media.image'),
              item.bytes ? ' · ' + formatBytes(item.bytes) : ''
            ))))),
        filtered.length > limit
          ? h('button', {
            type: 'button',
            className: 'dsh-nt-media-more',
            onClick: () => setLimit(value => value + 24),
          }, t('media.more', { count: filtered.length - limit }))
          : null
      )
    }

    const MAX_TEXT_BYTES = 512 * 1024

    function useTextFile(file) {
      const [state, setState] = useState({ status: file ? 'loading' : 'idle', text: '', bytes: 0, truncated: false, error: null })
      useEffect(() => {
        if (!file || !file.path) {
          setState({ status: 'idle', text: '', bytes: 0, truncated: false, error: null })
          return undefined
        }
        let cancelled = false
        const controller = new AbortController()
        setState({ status: 'loading', text: '', bytes: 0, truncated: false, error: null })
        fetch(mediaFileUrl(file.path), { credentials: 'same-origin', signal: controller.signal })
          .then(async response => {
            if (!response.ok) throw new Error('HTTP ' + response.status)
            const headerLength = Number(response.headers.get('content-length') || 0)
            if (Number.isFinite(headerLength) && headerLength > MAX_TEXT_BYTES) {
              const tooLarge = new Error('too-large')
              tooLarge.tooLarge = true
              tooLarge.bytes = headerLength
              if (response.body) response.body.cancel().catch(() => {})
              throw tooLarge
            }
            if (!response.body || typeof response.body.getReader !== 'function') {
              const text = await response.text()
              return { text: text.slice(0, MAX_TEXT_BYTES), bytes: text.length, truncated: text.length > MAX_TEXT_BYTES }
            }
            const reader = response.body.getReader()
            const chunks = []
            let total = 0
            let truncated = false
            for (;;) {
              const { done, value } = await reader.read()
              if (done) break
              if (!value || !value.byteLength) continue
              total += value.byteLength
              if (total > MAX_TEXT_BYTES) {
                const remaining = MAX_TEXT_BYTES - (total - value.byteLength)
                if (remaining > 0) chunks.push(value.subarray(0, remaining))
                truncated = true
                try { await reader.cancel() } catch {}
                break
              }
              chunks.push(value)
            }
            let length = 0
            for (const chunk of chunks) length += chunk.byteLength
            const merged = new Uint8Array(length)
            let offset = 0
            for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.byteLength }
            return { text: new TextDecoder('utf-8', { fatal: false }).decode(merged), bytes: total, truncated }
          })
          .then(result => {
            if (cancelled) return
            setState({ status: 'ready', text: result.text, bytes: result.bytes, truncated: result.truncated, error: null })
          })
          .catch(error => {
            if (cancelled || (error && error.name === 'AbortError')) return
            if (error && error.tooLarge) {
              setState({ status: 'too-large', text: '', bytes: error.bytes || 0, truncated: false, error: null })
              return
            }
            setState({ status: 'error', text: '', bytes: 0, truncated: false, error: error || 'failed' })
          })
        return () => { cancelled = true; controller.abort() }
      }, [file && file.id, file && file.path])
      return state
    }

    function renderInlineMarkdown(text) {
      const nodes = []
      const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*|\[[^\]\n]+\]\(https?:\/\/[^)\s]+\))/g
      let last = 0
      let match
      while ((match = pattern.exec(text)) !== null) {
        if (match.index > last) nodes.push(text.slice(last, match.index))
        const token = match[0]
        const key = 'inline-' + match.index
        if (token.startsWith('`')) {
          nodes.push(h('code', { key }, token.slice(1, -1)))
        } else if (token.startsWith('**')) {
          nodes.push(h('strong', { key }, token.slice(2, -2)))
        } else if (token.startsWith('*')) {
          nodes.push(h('em', { key }, token.slice(1, -1)))
        } else {
          const link = /^\[([^\]]+)\]\((https?:\/\/[^)]+)\)$/.exec(token)
          const imageSyntax = match.index > 0 && text[match.index - 1] === '!'
          if (link && !imageSyntax) {
            nodes.push(h('a', {
              key,
              href: link[2],
              target: '_blank',
              rel: 'noreferrer noopener',
            }, link[1]))
          } else {
            nodes.push(token)
          }
        }
        last = pattern.lastIndex
      }
      if (last < text.length) nodes.push(text.slice(last))
      return nodes
    }

    function splitMarkdownRow(line) {
      return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim())
    }

    function isMarkdownTableSeparator(line) {
      return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line)
    }

    /** Minimal, HTML-free Markdown rendering. Raw HTML stays text; images are never fetched. */
    function renderMarkdown(text) {
      const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n').slice(0, 3000)
      const blocks = []
      let index = 0
      let key = 0
      const inline = value => renderInlineMarkdown(value)
      while (index < lines.length) {
        const line = lines[index]
        if (!line.trim()) { index += 1; continue }
        const fence = /^```([\w-]*)\s*$/.exec(line)
        if (fence) {
          index += 1
          const code = []
          while (index < lines.length && !/^```\s*$/.test(lines[index])) {
            code.push(lines[index])
            index += 1
          }
          if (index < lines.length) index += 1
          blocks.push(h('pre', { key: 'md-' + key++ }, h('code', null, code.join('\n'))))
          continue
        }
        const heading = /^(#{1,6})\s+(.*)$/.exec(line)
        if (heading) {
          const level = Math.min(6, heading[1].length)
          blocks.push(h('h' + level, { key: 'md-' + key++ }, inline(heading[2])))
          index += 1
          continue
        }
        if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {
          blocks.push(h('hr', { key: 'md-' + key++ }))
          index += 1
          continue
        }
        if (line.includes('|') && index + 1 < lines.length && isMarkdownTableSeparator(lines[index + 1])) {
          const header = splitMarkdownRow(line).slice(0, 12)
          index += 2
          const rows = []
          while (index < lines.length && lines[index].includes('|') && lines[index].trim() && rows.length < 200) {
            rows.push(splitMarkdownRow(lines[index]).slice(0, header.length))
            index += 1
          }
          blocks.push(h('div', { className: 'dsh-nt-md-table-wrap', key: 'md-' + key++ },
            h('table', { className: 'dsh-nt-md-table' },
              h('thead', null, h('tr', null, header.map((cell, cellIndex) => h('th', { key: cellIndex }, inline(cell))))),
              h('tbody', null, rows.map((row, rowIndex) => h('tr', { key: rowIndex },
                header.map((_, cellIndex) => h('td', { key: cellIndex }, inline(row[cellIndex] || '')))))))))
          continue
        }
        if (/^>\s?/.test(line)) {
          const quote = []
          while (index < lines.length && /^>\s?/.test(lines[index])) {
            quote.push(lines[index].replace(/^>\s?/, ''))
            index += 1
          }
          blocks.push(h('blockquote', { key: 'md-' + key++ }, inline(quote.join(' '))))
          continue
        }
        if (/^\s*[-*+]\s+/.test(line)) {
          const items = []
          while (index < lines.length && /^\s*[-*+]\s+/.test(lines[index])) {
            items.push(lines[index].replace(/^\s*[-*+]\s+/, ''))
            index += 1
          }
          blocks.push(h('ul', { key: 'md-' + key++ }, items.map((item, itemIndex) => h('li', { key: itemIndex }, inline(item)))))
          continue
        }
        if (/^\s*\d+[.)]\s+/.test(line)) {
          const items = []
          while (index < lines.length && /^\s*\d+[.)]\s+/.test(lines[index])) {
            items.push(lines[index].replace(/^\s*\d+[.)]\s+/, ''))
            index += 1
          }
          blocks.push(h('ol', { key: 'md-' + key++ }, items.map((item, itemIndex) => h('li', { key: itemIndex }, inline(item)))))
          continue
        }
        const paragraph = []
        while (index < lines.length && lines[index].trim()
          && !/^```/.test(lines[index])
          && !/^(#{1,6})\s+/.test(lines[index])
          && !/^\s*([-*_])\1{2,}\s*$/.test(lines[index])
          && !/^>\s?/.test(lines[index])
          && !/^\s*[-*+]\s+/.test(lines[index])
          && !/^\s*\d+[.)]\s+/.test(lines[index])) {
          paragraph.push(lines[index])
          index += 1
        }
        if (paragraph.length) blocks.push(h('p', { key: 'md-' + key++ }, inline(paragraph.join(' '))))
      }
      return blocks
    }

    function parseDelimited(text, delimiter) {
      const rows = []
      let row = []
      let cell = ''
      let quoted = false
      let truncated = false
      const source = String(text || '').replace(/\r\n?/g, '\n')
      for (let index = 0; index < source.length; index += 1) {
        const char = source[index]
        if (quoted) {
          if (char === '"') {
            if (source[index + 1] === '"') { cell += '"'; index += 1 }
            else quoted = false
          } else {
            cell += char
          }
          continue
        }
        if (char === '"') { quoted = true; continue }
        if (char === delimiter) {
          row.push(cell)
          cell = ''
          if (row.length >= 30) { truncated = true; row = row.slice(0, 30) }
          continue
        }
        if (char === '\n') {
          row.push(cell)
          cell = ''
          rows.push(row)
          row = []
          if (rows.length >= 200) { truncated = true; break }
          continue
        }
        cell += char
      }
      if (!truncated && (cell || row.length)) {
        row.push(cell)
        rows.push(row)
      }
      return { rows, truncated }
    }

    function renderDelimitedTable(text, delimiter) {
      const parsed = parseDelimited(text, delimiter)
      const rows = parsed.rows
      if (!rows.length) return h('pre', { className: 'dsh-nt-text-pre' }, text)
      const header = rows[0]
      const body = rows.slice(1)
      const headRow = h('tr', null, header.map((cell, index) => h('th', { key: index }, cell)))
      const bodyRows = body.map((row, rowIndex) => h('tr', { key: rowIndex },
        header.map((_, cellIndex) => h('td', { key: cellIndex }, row[cellIndex] || ''))))
      return h('div', { className: 'dsh-nt-csv-wrap' },
        h('table', { className: 'dsh-nt-csv-table' },
          h('thead', null, headRow),
          h('tbody', null, bodyRows)),
        parsed.truncated ? h('div', { className: 'dsh-nt-text-note' }, '…') : null)
    }

    function TextViewer(props) {
      const { file, t, locale, onClose, onSelectNode } = props
      const state = useTextFile(file)
      const [mode, setMode] = useState(file && (file.ext === 'md' || file.ext === 'markdown' || file.ext === 'mdx') ? 'preview' : file && (file.ext === 'csv' || file.ext === 'tsv') ? 'table' : 'text')
      const [copied, setCopied] = useState(null)
      const textStageRef = useRef(null)

      useEffect(() => {
        const onWheel = event => {
          const stage = textStageRef.current
          if (!stage || !event.target || !stage.contains(event.target)) return
          const line = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? stage.clientHeight : 1
          const x = (event.shiftKey ? event.deltaY : event.deltaX) * line
          const y = (event.shiftKey ? 0 : event.deltaY) * line
          const maxTop = stage.scrollHeight - stage.clientHeight
          const maxLeft = stage.scrollWidth - stage.clientWidth
          const canY = maxTop > 0 && y !== 0
          const canX = maxLeft > 0 && x !== 0
          if (!canY && !canX) return
          event.preventDefault()
          event.stopPropagation()
          if (canY) stage.scrollTop = Math.max(0, Math.min(maxTop, stage.scrollTop + y))
          if (canX) stage.scrollLeft = Math.max(0, Math.min(maxLeft, stage.scrollLeft + x))
        }
        window.addEventListener('wheel', onWheel, { passive: false, capture: true })
        return () => window.removeEventListener('wheel', onWheel, true)
      }, [file && file.id, mode])
      const isMarkdown = file && (file.ext === 'md' || file.ext === 'markdown' || file.ext === 'mdx')
      const isDelimited = file && (file.ext === 'csv' || file.ext === 'tsv')

      useEffect(() => {
        setMode(isMarkdown ? 'preview' : isDelimited ? 'table' : 'text')
      }, [file && file.id, isMarkdown, isDelimited])

      useEffect(() => {
        const onKeyDown = event => {
          if (event.key !== 'Escape') return
          event.stopImmediatePropagation()
          event.preventDefault()
          onClose()
        }
        document.addEventListener('keydown', onKeyDown, true)
        return () => document.removeEventListener('keydown', onKeyDown, true)
      }, [onClose])

      useEffect(() => {
        if (!copied) return undefined
        const timer = setTimeout(() => setCopied(null), 1600)
        return () => clearTimeout(timer)
      }, [copied])

      function copyText() {
        try {
          if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
            navigator.clipboard.writeText(state.text).then(() => setCopied('ok')).catch(() => setCopied('fail'))
            return
          }
          setCopied('fail')
        } catch { setCopied('fail') }
      }

      const lines = state.text ? state.text.split('\n').length : 0
      const size = state.bytes ? formatBytes(state.bytes) : ''
      const copyLabel = copied === 'ok' ? t('file.copied') : copied === 'fail' ? t('file.copyFailed') : t('file.copy')

      return h('div', { className: 'dsh-nt-media-viewer dsh-nt-text-viewer', role: 'dialog', 'aria-modal': 'true', 'aria-label': file.name },
        h('button', { type: 'button', className: 'dsh-nt-media-viewer-backdrop', onClick: onClose, 'aria-label': t('file.close') }),
        h('div', { className: 'dsh-nt-media-viewer-card', 'data-nt-scroll': 'true' },
          h('div', { className: 'dsh-nt-media-viewer-head' },
            h('span', { className: 'dsh-nt-media-viewer-title' }, file.name),
            h('button', { type: 'button', className: 'dsh-nt-icon-button', title: t('file.close'), 'aria-label': t('file.close'), onClick: onClose }, h(IconClose))),
          h('div', { className: 'dsh-nt-text-toolbar' },
            isMarkdown ? h('div', { className: 'dsh-nt-text-modes' },
              h('button', { type: 'button', className: 'dsh-nt-media-filter' + (mode === 'preview' ? ' is-active' : ''), onClick: () => setMode('preview') }, t('file.preview')),
              h('button', { type: 'button', className: 'dsh-nt-media-filter' + (mode === 'text' ? ' is-active' : ''), onClick: () => setMode('text') }, t('file.text'))) : null,
            isDelimited ? h('div', { className: 'dsh-nt-text-modes' },
              h('button', { type: 'button', className: 'dsh-nt-media-filter' + (mode === 'table' ? ' is-active' : ''), onClick: () => setMode('table') }, t('file.table')),
              h('button', { type: 'button', className: 'dsh-nt-media-filter' + (mode === 'text' ? ' is-active' : ''), onClick: () => setMode('text') }, t('file.text'))) : null,
            h('span', { className: 'dsh-nt-text-toolbar-spacer' }),
            h('button', { type: 'button', className: 'dsh-nt-ghost', onClick: copyText }, h(IconCopy), h('span', null, copyLabel)),
            h('a', { className: 'dsh-nt-ghost', href: mediaFileUrl(file.path), download: file.name }, h(IconDownload), h('span', null, t('file.download')))),
          h('div', { className: 'dsh-nt-media-viewer-stage dsh-nt-text-stage', 'data-nt-scroll': 'true', tabIndex: 0, role: 'region', 'aria-label': file.name, ref: textStageRef },
            state.status === 'loading'
              ? h('span', { className: 'dsh-nt-media-viewer-status' }, h(IconSpinner), t('file.loading'))
              : state.status === 'too-large'
                ? h('span', { className: 'dsh-nt-media-viewer-status is-error' }, h(IconAlert), t('file.tooLarge', { size: formatBytes(state.bytes) }))
                : state.status === 'error'
                  ? h('span', { className: 'dsh-nt-media-viewer-status is-error' }, h(IconAlert), t('file.loadFailed'))
                  : mode === 'preview' && isMarkdown
                    ? h('div', { className: 'dsh-nt-md' }, renderMarkdown(state.text))
                    : mode === 'table' && isDelimited
                      ? renderDelimitedTable(state.text, file.ext === 'tsv' ? '\t' : ',')
                      : h('pre', { className: 'dsh-nt-text-pre' }, state.text)),
          state.status === 'ready' && state.truncated
            ? h('div', { className: 'dsh-nt-text-note' }, t('file.truncated', { size: formatBytes(MAX_TEXT_BYTES) }))
            : null,
          h('div', { className: 'dsh-nt-media-viewer-meta' },
            h('div', { className: 'dsh-nt-media-meta-row' }, h('span', { className: 'dsh-nt-media-meta-key' }, t('file.type')), h('span', { className: 'dsh-nt-media-meta-value' }, file.ext.toUpperCase())),
            h('div', { className: 'dsh-nt-media-meta-row' }, h('span', { className: 'dsh-nt-media-meta-key' }, t('media.source')), h('span', { className: 'dsh-nt-media-meta-value' }, file.toolName || file.source || '—')),
            h('div', { className: 'dsh-nt-media-meta-row' }, h('span', { className: 'dsh-nt-media-meta-key' }, t('media.time')), h('span', { className: 'dsh-nt-media-meta-value' }, timeAgo(file.time, locale, t))),
            size ? h('div', { className: 'dsh-nt-media-meta-row' }, h('span', { className: 'dsh-nt-media-meta-key' }, t('media.size')), h('span', { className: 'dsh-nt-media-meta-value' }, size)) : null,
            lines ? h('div', { className: 'dsh-nt-media-meta-row' }, h('span', { className: 'dsh-nt-media-meta-key' }, t('file.lines')), h('span', { className: 'dsh-nt-media-meta-value' }, String(lines))) : null),
          h('div', { className: 'dsh-nt-media-viewer-path', title: file.path }, file.path),
          h('div', { className: 'dsh-nt-media-viewer-actions' },
            file.nodeId ? h('button', { type: 'button', className: 'dsh-nt-primary', onClick: () => { onClose(); onSelectNode(file.nodeId) } }, h(IconArrowRight), h('span', null, t('media.openNode'))) : null,
            h('a', { className: 'dsh-nt-ghost', href: mediaFileUrl(file.path), download: file.name }, h(IconDownload), h('span', null, t('media.download'))))))
    }

    function FilesPanel(props) {
      const {
        files, t, locale, selectedId, onOpen, onClose, onSelectNode,
        dropped, hasMore, loadingOlder, onLoadOlder,
      } = props
      const items = Array.isArray(files) ? files : []
      const [filter, setFilter] = useState('all')
      const groups = ['docs', 'data', 'code']
      const counts = { docs: 0, data: 0, code: 0 }
      for (const item of items) if (counts[item.group] !== undefined) counts[item.group] += 1
      const filtered = filter === 'all' ? items : items.filter(item => item.group === filter)
      const selected = items.find(item => item.id === selectedId) || null

      useEffect(() => {
        if (selectedId && !selected) onClose()
      }, [selectedId, selected, onClose])

      if (!items.length) {
        return h('div', { className: 'dsh-nt-media-empty' },
          h('span', { className: 'dsh-nt-media-empty-icon' }, h(IconDoc)),
          h('span', { className: 'dsh-nt-media-empty-title' }, t('file.empty')),
          h('span', { className: 'dsh-nt-media-empty-hint' }, t('file.emptyHint')),
          hasMore || loadingOlder
            ? h('button', { type: 'button', className: 'dsh-nt-media-more is-history', disabled: !!loadingOlder, onClick: () => { if (onLoadOlder) onLoadOlder() } },
              loadingOlder ? h(IconSpinner) : h(IconRefresh),
              h('span', null, loadingOlder ? t('file.loadingOlder') : t('file.loadOlder')))
            : null)
      }

      const fileRows = filtered.map(file => h('button', {
        key: file.id,
        type: 'button',
        className: 'dsh-nt-file-row' + (selectedId === file.id ? ' is-active' : ''),
        onClick: () => onOpen(file.id),
        title: file.path,
      },
      h('span', { className: 'dsh-nt-file-icon' }, h(IconDoc)),
      h('span', { className: 'dsh-nt-file-body' },
        h('span', { className: 'dsh-nt-file-name' }, file.name),
        h('span', { className: 'dsh-nt-file-meta' }, file.ext.toUpperCase(), ' · ', file.toolName || (file.source === 'deliverable' ? t('media.deliverable') : t('media.source')), ' · ', timeAgo(file.time, locale, t)),
        file.description ? h('span', { className: 'dsh-nt-file-desc' }, file.description) : null),
      h('span', { className: 'dsh-nt-file-arrow', 'aria-hidden': 'true' }, h(IconArrowRight))))

      const toolbar = h('div', { className: 'dsh-nt-media-toolbar' },
        h('div', { className: 'dsh-nt-media-filters' },
          h('button', { type: 'button', className: 'dsh-nt-media-filter' + (filter === 'all' ? ' is-active' : ''), onClick: () => setFilter('all') }, t('file.all'), h('span', { className: 'dsh-nt-media-filter-count' }, items.length)),
          groups.map(group => counts[group]
            ? h('button', { key: group, type: 'button', className: 'dsh-nt-media-filter' + (filter === group ? ' is-active' : ''), onClick: () => setFilter(group) },
              t('file.' + group), h('span', { className: 'dsh-nt-media-filter-count' }, counts[group]))
            : null)),
        dropped ? h('span', { className: 'dsh-nt-media-dropped', title: t('file.droppedHint') }, t('file.dropped', { count: dropped })) : null)

      const history = hasMore || loadingOlder
        ? h('button', { type: 'button', className: 'dsh-nt-media-more is-history', disabled: !!loadingOlder, onClick: () => { if (onLoadOlder) onLoadOlder() } },
          loadingOlder ? h(IconSpinner) : h(IconRefresh),
          h('span', null, loadingOlder ? t('file.loadingOlder') : t('file.loadOlder')))
        : null

      return h('div', { className: 'dsh-nt-files' },
        toolbar,
        history,
        h('div', { className: 'dsh-nt-file-list' }, fileRows))
    }

    function SidePanel(props) {
      const {
        runtime, snapshot, ui, uiStore, t, locale, onSelect, onFocusNode, onOpenSession,
        loadMedia, onOpenMedia, onOpenFile,
      } = props
      const projectMode = snapshot.mode === 'project'
      let tab = ui.tab
      if (projectMode) {
        if (tab !== 'chats' && tab !== 'node') tab = 'chats'
        if (tab === 'node' && !snapshot.selected) tab = 'chats'
      } else {
        if (tab !== 'flow' && tab !== 'signals' && tab !== 'calls' && tab !== 'media' && tab !== 'files' && tab !== 'node') tab = 'flow'
        if (tab === 'node' && !snapshot.selected) tab = 'flow'
      }
      const bodyRef = useRef(null)
      const stickRef = useRef(true)
      const [resizing, setResizing] = useState(false)
      const resizeRef = useRef(null)

      useEffect(() => () => {
        if (document.body) document.body.style.userSelect = ''
      }, [])

      function applySideWidth(width) {
        uiStore.patch({ sideWidth: sideWidthValue(width) })
      }

      function resizeStart(event) {
        if (event.button !== 0) return
        event.preventDefault()
        try { event.currentTarget.setPointerCapture(event.pointerId) } catch {}
        resizeRef.current = {
          startX: event.clientX,
          startWidth: sideWidthValue(uiStore.getSnapshot().sideWidth),
        }
        if (document.body) document.body.style.userSelect = 'none'
        setResizing(true)
      }

      function resizeMove(event) {
        const active = resizeRef.current
        if (!active) return
        applySideWidth(active.startWidth + (active.startX - event.clientX))
      }

      function resizeEnd(event) {
        if (!resizeRef.current) return
        resizeRef.current = null
        if (document.body) document.body.style.userSelect = ''
        try {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
        } catch {}
        setResizing(false)
        saveSideWidth(uiStore.getSnapshot().sideWidth)
      }

      function resizeKey(event) {
        const current = sideWidthValue(uiStore.getSnapshot().sideWidth)
        let next = null
        if (event.key === 'ArrowLeft') next = current + 16
        else if (event.key === 'ArrowRight') next = current - 16
        else if (event.key === 'Home') next = SIDE_DEFAULT
        if (next === null) return
        event.preventDefault()
        applySideWidth(next)
        saveSideWidth(next)
      }

      function resizeReset() {
        applySideWidth(SIDE_DEFAULT)
        saveSideWidth(SIDE_DEFAULT)
      }

      useEffect(() => {
        if (tab !== 'flow') return
        const element = bodyRef.current
        if (!element || !stickRef.current) return
        element.scrollTop = element.scrollHeight
      }, [snapshot.revision, tab])

      useEffect(() => {
        stickRef.current = true
        const current = uiStore.getSnapshot()
        if (current.filter) uiStore.patch({ filter: '' })
        if (current.mediaId) uiStore.patch({ mediaId: null })
        if (current.fileId) uiStore.patch({ fileId: null })
      }, [snapshot.sessionId, snapshot.mode, uiStore])

      function onScroll(event) {
        const element = event.currentTarget
        stickRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < 48
      }

      const tabs = projectMode
        ? [
          { key: 'chats', label: t('tabs.chats') },
          { key: 'node', label: t('tabs.node'), disabled: !snapshot.selected },
        ]
        : [
          { key: 'flow', label: t('tabs.flow') },
          { key: 'signals', label: t('tabs.signals') },
          { key: 'calls', label: t('tabs.calls'), count: snapshot.callCounts ? snapshot.callCounts.total : (snapshot.calls || []).length },
          { key: 'media', label: t('tabs.media'), count: (snapshot.media || []).length },
          { key: 'files', label: t('tabs.files'), count: (snapshot.files || []).length },
          { key: 'node', label: t('tabs.node'), disabled: !snapshot.selected },
        ]

      const showFilter = !projectMode && (tab === 'flow' || tab === 'signals' || tab === 'calls')
      const project = snapshot.project

      return h('aside', { className: 'dsh-nt-side' },
        h('div', {
          className: 'dsh-nt-resizer' + (resizing ? ' is-dragging' : ''),
          role: 'separator',
          'aria-orientation': 'vertical',
          'aria-label': t('panel.resizeSide'),
          title: t('panel.resizeSide'),
          tabIndex: 0,
          onPointerDown: resizeStart,
          onPointerMove: resizeMove,
          onPointerUp: resizeEnd,
          onPointerCancel: resizeEnd,
          onDoubleClick: resizeReset,
          onKeyDown: resizeKey,
        }),
        h('div', { className: 'dsh-nt-tabs', role: 'tablist' },
          tabs.map(item => h('button', {
            key: item.key,
            type: 'button',
            role: 'tab',
            className: 'dsh-nt-tab' + (tab === item.key ? ' is-active' : ''),
            'aria-selected': tab === item.key,
            disabled: !!item.disabled,
            onClick: () => uiStore.patch({ tab: item.key, mediaId: null, fileId: null }),
          }, item.label, item.key === 'signals' && snapshot.signals.length
            ? h('span', { className: 'dsh-nt-tab-count' }, snapshot.signals.length)
            : (item.key === 'calls' || item.key === 'media' || item.key === 'files') && item.count
              ? h('span', { className: 'dsh-nt-tab-count' }, item.count)
              : null))
        ),
        showFilter ? h('div', { className: 'dsh-nt-filter' },
          h(IconSearch),
          h('input', {
            value: ui.filter,
            placeholder: t('filter.placeholder'),
            'aria-label': t('filter.placeholder'),
            onChange: event => uiStore.patch({ filter: event.target.value }),
          }),
          ui.filter ? h('button', {
            type: 'button',
            className: 'dsh-nt-filter-clear',
            title: t('filter.clear'),
            'aria-label': t('filter.clear'),
            onClick: () => uiStore.patch({ filter: '' }),
          }, h(IconClose)) : null
        ) : null,
        h('div', {
          className: 'dsh-nt-side-body',
          ref: bodyRef,
          onScroll,
          role: 'tabpanel',
          'data-nt-scroll': 'true',
        },
        projectMode
          ? tab === 'chats'
            ? h(ProjectPanel, { project, snapshot, t, locale, onOpenSession })
            : h(NodePanel, {
              snapshot, t, locale, onSelect, onFocusNode, onOpenSession,
              loadMedia, onOpenMedia, onOpenFile,
              onBack: () => uiStore.patch({ tab: 'chats' }),
            })
          : tab === 'flow'
            ? h(FlowPanel, {
              snapshot, t, locale, onSelect,
              live: snapshot.running,
              flowLimit: ui.flowLimit,
              filter: ui.filter,
              onMore: () => uiStore.patch({ flowLimit: 120 }),
            })
            : tab === 'signals'
              ? h(SignalsPanel, { snapshot, t, locale, onSelect, filter: ui.filter })
              : tab === 'calls'
                ? h(CallsPanel, {
                  snapshot,
                  t,
                  locale,
                  onSelect,
                  filter: ui.filter,
                  selectedId: snapshot.selectedId,
                  onClearFilter: () => uiStore.patch({ filter: '' }),
                })
                : tab === 'media'
                  ? h(MediaPanel, {
                  media: snapshot.media,
                  sessionId: snapshot.sessionId,
                  loadMedia,
                  t,
                  locale,
                  selectedId: ui.mediaId,
                  dropped: snapshot.mediaDropped,
                  hasMore: snapshot.hasMore,
                  loadingOlder: snapshot.loadingOlder,
                  onLoadOlder: () => runtime.loadOlder(),
                  onOpen: id => uiStore.patch({ mediaId: id }),
                  onClose: () => uiStore.patch({ mediaId: null }),
                  onSelectNode: onSelect,
                })
                : tab === 'files'
                  ? h(FilesPanel, {
                    files: snapshot.files,
                    t,
                    locale,
                    selectedId: ui.fileId,
                    dropped: snapshot.filesDropped,
                    hasMore: snapshot.hasMore,
                    loadingOlder: snapshot.loadingOlder,
                    onLoadOlder: () => runtime.loadOlder(),
                    onOpen: id => uiStore.patch({ fileId: id }),
                    onClose: () => uiStore.patch({ fileId: null }),
                    onSelectNode: onSelect,
                  })
                  : h(NodePanel, {
                    snapshot, t, locale, onSelect, onFocusNode, onOpenSession,
                    loadMedia, onOpenMedia, onOpenFile,
                    onBack: () => uiStore.patch({ tab: 'flow' }),
                  })
        ),
        h('div', { className: 'dsh-nt-side-foot' },
          projectMode && project
            ? h('span', null, t('project.chatCount', { count: project.total }))
            : h('span', null, t('session.events', { count: snapshot.eventCount || 0 })),
          h('span', { className: 'dsh-nt-foot-sep' }, '·'),
          projectMode && project
            ? h('span', null, t('project.connections') + ': ' + project.connections)
            : h('span', null, t('session.toolCount', { count: snapshot.toolCount || 0 })),
          projectMode && project && project.running
            ? h('span', { className: 'dsh-nt-foot-sep' }, '·')
            : null,
          projectMode && project && project.running
            ? h('span', null, t('session.runningCount', { count: project.running }))
            : null,
          !projectMode && snapshot.tokenTotal
            ? h('span', { className: 'dsh-nt-foot-sep' }, '·')
            : null,
          !projectMode && snapshot.tokenTotal
            ? h('span', null, t('model.tokens', { n: formatTokens(snapshot.tokenTotal) }))
            : null
        )
      )
    }

    function ProjectPanel(props) {
      const { project, t, locale, onOpenSession } = props
      const [query, setQuery] = useState('')
      if (!project) {
        return h('div', { className: 'dsh-nt-empty-panel' },
          h('div', { className: 'dsh-nt-empty-orb' }, h(BrainMark, { size: 32 })),
          h('strong', null, t('project.noProject'))
        )
      }
      const chats = Array.isArray(project.chats) ? project.chats : []
      const needle = query.trim().toLowerCase()
      const filtered = chats
        .filter(chat => {
          if (!needle) return true
          return String(chat.title || '').toLowerCase().includes(needle)
            || String(chat.cwd || '').toLowerCase().includes(needle)
        })
        .slice()
        .sort((a, b) => {
          if (!!a.running !== !!b.running) return a.running ? -1 : 1
          return (b.updatedAt || 0) - (a.updatedAt || 0)
        })
      const parentTitles = new Map()
      for (const chat of chats) parentTitles.set(chat.id, chat.title)
      return h('div', { className: 'dsh-nt-project' },
        h('div', { className: 'dsh-nt-project-summary' },
          h('div', { className: 'dsh-nt-project-title' }, project.title),
          h('div', { className: 'dsh-nt-project-path', title: project.path }, project.path),
          h('div', { className: 'dsh-nt-project-stats' },
            h('span', { className: 'dsh-nt-project-stat' }, t('project.chatCount', { count: project.total })),
            h('span', { className: 'dsh-nt-project-stat is-links' }, t('project.connections') + ': ' + project.connections),
            project.running
              ? h('span', { className: 'dsh-nt-project-stat is-running' },
                h('span', { className: 'dsh-nt-live-dot', 'aria-hidden': 'true' }),
                t('session.runningCount', { count: project.running }))
              : null
          )
        ),
        h('div', { className: 'dsh-nt-filter dsh-nt-project-search' },
          h(IconSearch),
          h('input', {
            value: query,
            placeholder: t('project.search'),
            'aria-label': t('project.search'),
            onChange: event => setQuery(event.target.value),
          }),
          query ? h('button', {
            type: 'button',
            className: 'dsh-nt-filter-clear',
            title: t('filter.clear'),
            'aria-label': t('filter.clear'),
            onClick: () => setQuery(''),
          }, h(IconClose)) : null
        ),
        filtered.length ? h('div', { className: 'dsh-nt-project-list' },
          filtered.map(chat => {
            const parentTitle = chat.parentId ? parentTitles.get(chat.parentId) : null
            return h('button', {
              key: chat.id,
              type: 'button',
              className: 'dsh-nt-project-row' + (chat.linked ? ' is-linked' : '') + (chat.shown ? '' : ' is-muted'),
              onClick: () => { if (onOpenSession) onOpenSession(chat.id) },
              title: t('project.open'),
            },
            h('span', { className: 'dsh-nt-session-state' + (chat.running ? ' is-running' : ''), 'aria-hidden': 'true' }),
            h('span', { className: 'dsh-nt-project-row-main' },
              h('span', { className: 'dsh-nt-project-row-title' }, chat.title),
              h('span', { className: 'dsh-nt-project-row-meta' },
                chat.running ? t('session.running') : timeAgo(chat.updatedAt, locale, t),
                chat.linked ? h('span', { className: 'dsh-nt-project-badge' }, t('project.legendLink')) : null,
                parentTitle ? h('span', { className: 'dsh-nt-project-parent' }, '↳ ' + clip(parentTitle, 28)) : null
              )
            ),
            h('span', { className: 'dsh-nt-project-open', 'aria-hidden': 'true' }, h(IconArrowRight))
            )
          })
        ) : h('div', { className: 'dsh-nt-empty-panel' },
          h('div', { className: 'dsh-nt-empty-orb' }, h(BrainMark, { size: 30 })),
          h('strong', null, needle ? t('filter.empty') : t('project.empty')),
          needle ? null : h('span', null, t('project.emptyHint'))
        )
      )
    }

    function FlowPanel(props) {
      const { snapshot, t, locale, onSelect, live, flowLimit, onMore, filter } = props
      const needle = String(filter || '').trim().toLowerCase()
      const allFlow = (snapshot.flow || []).filter(entry => {
        if (!needle) return true
        return String(entry.text || '').toLowerCase().includes(needle)
          || String(entry.title || '').toLowerCase().includes(needle)
          || kindLabel(entry.kind, t).toLowerCase().includes(needle)
          || sectorLabel(entry.sector, t, true).toLowerCase().includes(needle)
      })
      const limit = flowLimit || 40
      const flow = allFlow.length > limit ? allFlow.slice(-limit) : allFlow
      const anchors = snapshot.anchors || []

      const anchorStrip = anchors.length ? h('div', { className: 'dsh-nt-anchors' },
        anchors.map(anchor => {
          const color = anchor.color || (SECTORS[anchor.sector] || SECTORS.parietal).color
          return h('button', {
            key: anchor.key,
            type: 'button',
            className: 'dsh-nt-anchor',
            style: { '--nt-color': color },
            onClick: () => onSelect(anchor.nodeId),
            title: t('anchor.' + anchor.key),
          },
          h('span', { className: 'dsh-nt-anchor-dot', 'aria-hidden': 'true' }),
          h('span', null, t('anchor.' + anchor.key)),
          anchor.count > 1 ? h('span', { className: 'dsh-nt-anchor-count' }, anchor.count) : null
          )
        })
      ) : null

      if (!flow.length) {
        return h('div', { className: 'dsh-nt-flow-wrap' },
          anchorStrip,
          h('div', { className: 'dsh-nt-empty-panel' },
            h('div', { className: 'dsh-nt-empty-orb' }, h(BrainMark, { size: 34 })),
            h('strong', null, needle ? t('filter.empty') : t('flow.empty')),
            h('span', null, needle ? t('filter.clear') : t('flow.emptyHint'))
          )
        )
      }

      return h('div', { className: 'dsh-nt-flow-wrap' },
        anchorStrip,
        h('div', { className: 'dsh-nt-flow' },
          flow.map(entry => {
            const color = (SECTORS[entry.sector] || SECTORS.parietal).color
            return h('button', {
              key: entry.id,
              type: 'button',
              className: 'dsh-nt-flow-item' + (entry.live ? ' is-live' : ''),
              style: { '--nt-color': color },
              onClick: () => onSelect(entry.id),
              title: t('node.inspect'),
            },
            h('span', { className: 'dsh-nt-flow-rail', 'aria-hidden': 'true' }),
            h('span', { className: 'dsh-nt-flow-main' },
              h('span', { className: 'dsh-nt-flow-meta' },
                h('span', { className: 'dsh-nt-flow-kind' }, entry.live ? t('flow.live') : kindLabel(entry.kind, t)),
                h('span', { className: 'dsh-nt-flow-sector' }, sectorLabel(entry.sector, t, true)),
                h('span', { className: 'dsh-nt-flow-time' }, clockOf(entry.time, locale))
              ),
              h('span', { className: 'dsh-nt-flow-text' }, entry.text || '…',
                entry.live ? h('span', { className: 'dsh-nt-caret', 'aria-hidden': 'true' }) : null)
            )
            )
          }),
          allFlow.length > flow.length
            ? h('button', {
              type: 'button',
              className: 'dsh-nt-more',
              onClick: () => { if (onMore) onMore() },
            }, t('flow.more'))
            : null,
          live ? h('div', { className: 'dsh-nt-flow-live' },
            h('span', { className: 'dsh-nt-live-dot', 'aria-hidden': 'true' }),
            h('span', null, t('status.thinking'))
          ) : null
        )
      )
    }

    function SignalsPanel(props) {
      const { snapshot, t, locale, onSelect, filter } = props
      const needle = String(filter || '').trim().toLowerCase()
      const signals = (snapshot.signals || []).filter(signal => {
        if (!needle) return true
        return String(signal.toolName || '').toLowerCase().includes(needle)
          || String(signal.summary || '').toLowerCase().includes(needle)
          || kindLabel(signal.kind, t).toLowerCase().includes(needle)
          || sectorLabel(signal.sector, t, true).toLowerCase().includes(needle)
      })
      if (!signals.length) {
        return h('div', { className: 'dsh-nt-empty-panel' },
          h('div', { className: 'dsh-nt-empty-orb' }, h(IconBolt)),
          h('strong', null, needle ? t('filter.empty') : t('signals.empty')),
          needle ? h('span', null, t('filter.clear')) : null
        )
      }
      return h('div', { className: 'dsh-nt-signals' },
        signals.map(signal => {
          const color = (SECTORS[signal.sector] || SECTORS.parietal).color
          const status = signal.status === 'running' ? 'is-running' : signal.status === 'error' ? 'is-error' : 'is-done'
          return h('button', {
            key: signal.id,
            type: 'button',
            className: 'dsh-nt-signal ' + status,
            style: { '--nt-color': color },
            onClick: () => onSelect(signal.id),
            title: signal.summary || signal.toolName,
          },
          h('span', { className: 'dsh-nt-signal-icon' },
            signal.status === 'error' ? h(IconAlert) : signal.status === 'running' ? h(IconSpinner) : h(IconBolt)
          ),
          h('span', { className: 'dsh-nt-signal-main' },
            h('span', { className: 'dsh-nt-signal-name' }, signal.toolName),
            signal.summary ? h('span', { className: 'dsh-nt-signal-summary' }, signal.summary) : null
          ),
          h('span', { className: 'dsh-nt-signal-side' },
            h('span', { className: 'dsh-nt-signal-sector' }, sectorLabel(signal.sector, t, true)),
            h('span', { className: 'dsh-nt-signal-time' }, clockOf(signal.time, locale))
          )
          )
        })
      )
    }

    function CallsPanel(props) {
      const { snapshot, t, locale, onSelect, filter, selectedId, onClearFilter } = props
      const [family, setFamily] = useState('all')
      const calls = Array.isArray(snapshot.calls) ? snapshot.calls : []
      const counts = snapshot.callCounts || { skill: 0, mcp: 0, total: 0 }
      const needle = String(filter || '').trim().toLowerCase()

      if (!counts.total) {
        return h('div', { className: 'dsh-nt-empty-panel' },
          h('div', { className: 'dsh-nt-empty-orb' }, h(IconMcp)),
          h('strong', null, t('calls.empty')),
          h('span', null, t('calls.emptyHint'))
        )
      }

      const visible = calls.filter(call => {
        if (family !== 'all' && call.family !== family) return false
        if (!needle) return true
        return [call.toolName, call.skillName, call.mcpServer, call.mcpTool, call.title, call.summary]
          .some(value => String(value || '').toLowerCase().includes(needle))
      })
      const groups = ['skill', 'mcp']
        .filter(key => family === 'all' || key === family)
        .map(key => ({
          key,
          label: t(key === 'skill' ? 'calls.skill' : 'calls.mcp'),
          color: callFamilyColor(key),
          calls: visible.filter(call => call.family === key),
        }))
        .filter(group => group.calls.length)

      const chips = [
        { key: 'all', label: t('calls.all'), count: counts.total, color: '#b9d8f2' },
        { key: 'skill', label: t('calls.skill'), count: counts.skill, color: callFamilyColor('skill') },
        { key: 'mcp', label: t('calls.mcp'), count: counts.mcp, color: callFamilyColor('mcp') },
      ]

      return h('div', { className: 'dsh-nt-calls' },
        h('div', { className: 'dsh-nt-calls-head' },
          h('div', { className: 'dsh-nt-calls-heading' },
            h('strong', null, t('calls.title')),
            h('span', null, t('calls.subtitle'))
          ),
          h('span', { className: 'dsh-nt-calls-total' }, t('calls.total', { count: counts.total }))
        ),
        h('div', { className: 'dsh-nt-call-chips', role: 'tablist', 'aria-label': t('calls.title') },
          chips.map(chip => h('button', {
            key: chip.key,
            type: 'button',
            role: 'tab',
            'aria-selected': family === chip.key,
            className: 'dsh-nt-call-chip' + (family === chip.key ? ' is-active' : ''),
            style: { '--nt-color': chip.color },
            onClick: () => setFamily(chip.key),
          }, chip.key === 'all' ? h(IconLayers) : h(CallFamilyIcon, { family: chip.key }),
          h('span', null, chip.label),
          h('span', { className: 'dsh-nt-call-chip-count' }, chip.count)))
        ),
        !visible.length
          ? h('div', { className: 'dsh-nt-empty-panel is-compact' },
            h('div', { className: 'dsh-nt-empty-orb' }, h(IconSearch)),
            h('strong', null, t('calls.emptyFiltered')),
            needle ? h('button', { type: 'button', className: 'dsh-nt-call-clear', onClick: onClearFilter }, t('filter.clear')) : null
          )
          : groups.map(group => h('section', { key: group.key, className: 'dsh-nt-call-group' },
            h('div', { className: 'dsh-nt-call-group-head', style: { '--nt-color': group.color } },
              h('span', { className: 'dsh-nt-call-group-dot' }),
              h('span', null, group.label),
              h('span', { className: 'dsh-nt-call-group-count' }, group.calls.length)
            ),
            group.calls.map(call => {
              const color = callFamilyColor(call.family)
              const name = call.family === 'skill'
                ? (call.skillName || call.title || call.toolName)
                : [call.mcpServer || 'MCP', call.mcpTool || call.title].filter(Boolean).join(' / ')
              const detail = call.family === 'skill'
                ? (call.summary || t('calls.skillLoaded'))
                : (call.summary || '')
              const status = call.status === 'running' ? 'is-running' : call.status === 'error' ? 'is-error' : 'is-done'
              return h('button', {
                key: call.id,
                type: 'button',
                className: 'dsh-nt-call ' + status + (selectedId === call.id ? ' is-selected' : ''),
                style: { '--nt-color': color },
                onClick: () => onSelect(call.id),
                title: [call.toolName, call.summary].filter(Boolean).join('\n'),
              },
              h('span', { className: 'dsh-nt-call-icon' },
                call.status === 'error' ? h(IconAlert) : call.status === 'running' ? h(IconSpinner) : h(CallFamilyIcon, { family: call.family })
              ),
              h('span', { className: 'dsh-nt-call-main' },
                h('span', { className: 'dsh-nt-call-top' },
                  h('span', { className: 'dsh-nt-call-badge' }, t(call.family === 'skill' ? 'calls.skillTag' : 'calls.mcpTag')),
                  h('span', { className: 'dsh-nt-call-name' }, name)
                ),
                detail ? h('span', { className: 'dsh-nt-call-summary' }, detail) : null
              ),
              h('span', { className: 'dsh-nt-call-side' },
                h('span', { className: 'dsh-nt-call-time' }, clockOf(call.time, locale)),
                call.status === 'error' ? h('span', { className: 'dsh-nt-call-state' }, t('node.status.error')) : null
              )
              )
            })
          ))
      )
    }

    function NodePanel(props) {
      const {
        snapshot, t, locale, onSelect, onFocusNode, onOpenSession, onBack,
        loadMedia, onOpenMedia, onOpenFile,
      } = props
      const node = snapshot.selected
      const [copied, setCopied] = useState(null)

      useEffect(() => {
        if (!copied) return undefined
        const timer = setTimeout(() => setCopied(null), 1600)
        return () => clearTimeout(timer)
      }, [copied])

      if (!node) {
        return h('div', { className: 'dsh-nt-empty-panel' },
          h('div', { className: 'dsh-nt-empty-orb' }, h(BrainMark, { size: 32 })),
          h('strong', null, t('node.empty'))
        )
      }

      const family = node.family === 'skill' || node.family === 'mcp' ? node.family : null
      const color = family
        ? callFamilyColor(family)
        : (node.color || (SECTORS[node.sector] || SECTORS.parietal).color)
      const title = nodeTitle(node, t)
      const status = node.kind === 'session' || node.kind === 'subagent-session'
        ? (node.nodeRunning ? t('node.status.running') : t('node.status.done'))
        : statusLabel(node.status, t)
      const showArgs = !!node.argsRaw && node.kind !== 'reasoning' && node.kind !== 'answer'
      const showResult = !!node.result
      const showText = !!node.text && !showArgs && !showResult
      const meta = Array.isArray(node.meta) ? node.meta : []
      const inputs = Array.isArray(node.inputs) ? node.inputs : []
      const outputs = Array.isArray(node.outputs) ? node.outputs : []
      const media = Array.isArray(node.media) ? node.media : []
      const files = Array.isArray(node.files) ? node.files : []
      const project = snapshot.project
      const isProjectChat = !!node.sessionId
      const isProjectHub = node.kind === 'project'

      const metaRows = meta.slice()
      if (family === 'skill') {
        metaRows.unshift({ key: 'calls.skillTag', value: node.skillName || title })
      } else if (family === 'mcp') {
        metaRows.unshift(
          { key: 'calls.mcpServer', value: node.mcpServer || 'MCP' },
          { key: 'calls.mcpTool', value: node.mcpTool || node.title },
        )
      }
      if (isProjectChat) {
        metaRows.unshift({ key: 'node.status', value: status })
        if (node.origin === 'subagent') metaRows.push({ key: 'project.legendLink', value: t('node.kind.subagent-session') })
        if (node.parentSessionId) {
          const parent = project && Array.isArray(project.chats)
            ? project.chats.find(chat => chat.id === node.parentSessionId)
            : null
          metaRows.push({ key: 'project.parent', value: parent ? parent.title : node.parentSessionId })
        }
      }
      if (isProjectHub && project) {
        metaRows.push({ key: 'project.chatCountLabel', value: String(project.total) })
        metaRows.push({ key: 'project.connections', value: String(project.connections) })
      }

      function metaLabel(key) {
        const value = t(key)
        return value === key ? key.replace(/^node\./, '') : value
      }

      function copyContent() {
        const payload = [
          title,
          node.text,
          node.argsRaw ? prettyJson(node.argsRaw) : '',
          node.result,
        ].filter(Boolean).join('\n\n')
        try {
          if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
            navigator.clipboard.writeText(payload).then(() => setCopied('ok')).catch(() => setCopied('fail'))
            return
          }
          setCopied('fail')
        } catch {
          setCopied('fail')
        }
      }

      const copyLabel = copied === 'ok' ? t('node.copied') : copied === 'fail' ? t('node.copyFailed') : t('node.copy')

      function renderLinks(list, direction) {
        if (!list.length) {
          return h('div', { className: 'dsh-nt-links-empty' }, t('node.noLinks'))
        }
        return h('div', { className: 'dsh-nt-links' }, list.map(link => h('button', {
          key: link.id,
          type: 'button',
          className: 'dsh-nt-link-chip',
          style: { '--nt-color': link.color || (SECTORS[link.sector] || SECTORS.parietal).color },
          onClick: () => onSelect(link.id),
          title: nodeTitle(link, t),
        },
        h('span', { className: 'dsh-nt-link-arrow', 'aria-hidden': 'true' }, direction === 'in' ? '←' : '→'),
        h('span', { className: 'dsh-nt-link-dot', 'aria-hidden': 'true' }),
        clip(nodeTitle(link, t), 34)
        )))
      }

      return h('div', { className: 'dsh-nt-node' },
        h('div', { className: 'dsh-nt-node-hero', style: { '--nt-color': color } },
          h('span', { className: 'dsh-nt-node-orb', 'aria-hidden': 'true' }),
          h('div', { className: 'dsh-nt-node-hero-text' },
            h('span', { className: 'dsh-nt-node-kicker' },
              family
                ? h('span', { className: 'dsh-nt-node-family', style: { '--nt-color': color } },
                  h(CallFamilyIcon, { family }), t(family === 'skill' ? 'calls.skillTag' : 'calls.mcpTag'))
                : kindLabel(node.kind, t)
            ),
            h('h3', null, title),
            h('span', { className: 'dsh-nt-node-sub' },
              h('span', null, node.kind === 'project'
                ? t('project.legendHub')
                : node.kind === 'session'
                  ? t('project.legendChat')
                  : node.kind === 'subagent-session'
                    ? t('project.legendLink')
                    : sectorLabel(node.sector, t, false)),
              h('span', { className: 'dsh-nt-node-dot' }, '·'),
              clockOf(node.time, locale),
              h('span', { className: 'dsh-nt-node-dot' }, '·'),
              h('span', { className: 'dsh-nt-node-status ' + node.status }, status)
            )
          )
        ),

        node.synthetic ? h('div', { className: 'dsh-nt-node-note' }, t('node.titleDerived')) : null,
        node.error ? h('div', { className: 'dsh-nt-node-error' }, h(IconAlert), h('span', null, node.error)) : null,

        metaRows.length ? h('div', { className: 'dsh-nt-node-meta' },
          metaRows.map((entry, index) => h('div', { key: index, className: 'dsh-nt-node-meta-row' },
            h('span', null, metaLabel(entry.key)),
            h('span', { className: 'dsh-nt-node-meta-value' }, entry.value)
          ))
        ) : null,

        showArgs ? h(NodeSection, { title: t('node.args'), children: h('pre', { className: 'dsh-nt-pre', 'data-nt-scroll': 'true' }, prettyJson(node.argsRaw)) }) : null,
        showResult ? h(NodeSection, { title: t('node.result'), children: h('pre', { className: 'dsh-nt-pre', 'data-nt-scroll': 'true' }, node.result) }) : null,
        showText ? h(NodeSection, { title: kindLabel(node.kind, t), children: h('div', { className: 'dsh-nt-node-text', 'data-nt-scroll': 'true' }, node.text) }) : null,

        media.length && onOpenMedia
          ? h(NodeSection, {
            title: t('node.media'),
            children: h('div', { className: 'dsh-nt-node-media' }, media.map(item => h('button', {
              key: item.id,
              type: 'button',
              className: 'dsh-nt-node-media-tile',
              title: item.name,
              onClick: () => onOpenMedia(item),
            }, h(MediaThumb, {
              media: item,
              sessionId: snapshot.sessionId,
              loadMedia,
              alt: item.name,
              className: 'dsh-nt-node-media-image',
            }))))
          })
          : null,

        files.length && onOpenFile
          ? h(NodeSection, {
            title: t('node.files'),
            children: h('div', { className: 'dsh-nt-node-files' }, files.map(file => h('button', {
              key: file.id,
              type: 'button',
              className: 'dsh-nt-node-file',
              title: file.path,
              onClick: () => onOpenFile(file),
            }, h(IconDoc), h('span', { className: 'dsh-nt-node-file-name' }, file.name), h('span', { className: 'dsh-nt-node-file-ext' }, file.ext.toUpperCase())))),
          })
          : null,

        h(NodeSection, { title: t('node.inputs'), children: renderLinks(inputs, 'in') }),
        h(NodeSection, { title: t('node.outputs'), children: renderLinks(outputs, 'out') }),

        h('div', { className: 'dsh-nt-node-actions' },
          isProjectChat ? h('button', {
            type: 'button', className: 'dsh-nt-primary',
            onClick: () => { if (onOpenSession) onOpenSession(node.sessionId) },
          }, h(IconArrowRight), h('span', null, t('project.open'))) : null,
          h('button', {
            type: 'button', className: 'dsh-nt-primary',
            onClick: () => onFocusNode(node.id),
          }, h(IconOrbit), h('span', null, t('node.focus'))),
          h('button', {
            type: 'button',
            className: 'dsh-nt-ghost',
            onClick: copyContent,
            title: copyLabel,
          }, h(IconCopy), h('span', null, copyLabel)),
          h('button', {
            type: 'button', className: 'dsh-nt-ghost',
            onClick: onBack,
          }, t('node.close'))
        )
      )
    }

    function NodeSection(props) {
      return h('section', { className: 'dsh-nt-node-section' },
        h('h4', null, props.title),
        props.children
      )
    }

    /* ------------------------------------------------------------------ *
     *  Main panel
     * ------------------------------------------------------------------ */

    function NeuralPanelConnected(props) {
      const identity = useCallback(value => value, [])
      const workspaceSnapshot = props.useWorkspaces(identity)
      return h(NeuralPanel, Object.assign({}, props, { workspaceSnapshot }))
    }

    function NeuralPanel(props) {
      const { ctx, runtime, uiStore, workspaceSnapshot } = props
      const ui = useUi(uiStore)
      const snapshot = useRuntime(runtime)
      const list = useSessionList(ctx)
      const t = useT()
      const locale = useLang()
      const [refreshing, setRefreshing] = useState(false)
      const [focusRequest, setFocusRequest] = useState(null)
      const consoleRef = useRef(null)
      const previousFocusRef = useRef(null)

      const fallbackProjects = useMemo(() => buildProjects(list), [list])
      const workspaceItems = useMemo(() => {
        const registered = workspaceSnapshot && Array.isArray(workspaceSnapshot.items)
          ? workspaceSnapshot.items
          : []
        if (registered.length) {
          const byId = list && list.byId ? list.byId : {}
          const covered = new Set()
          for (const workspace of registered) {
            for (const id of (workspace.sessionIds || [])) covered.add(id)
          }
          const ids = list && Array.isArray(list.ids) ? list.ids : Object.keys(byId)
          const unassigned = ids.filter(id => {
            const row = byId[id]
            if (!row || row.blank) return false
            if (covered.has(id)) return false
            if (row.parentId && covered.has(row.parentId)) return false
            return true
          })
          if (!unassigned.length) return registered
          return registered.concat([{
            workspaceId: 'unassigned',
            title: t('project.noProject'),
            path: '',
            sessionIds: unassigned,
            createdAt: '',
            updatedAt: '',
          }])
        }
        return fallbackProjects.map(project => ({
          workspaceId: 'cwd:' + (project.key || ''),
          title: project.title || t('project.noProject'),
          path: project.path,
          sessionIds: project.chats.map(chat => chat.id),
          createdAt: '',
          updatedAt: project.updatedAt ? new Date(project.updatedAt).toISOString() : '',
        }))
      }, [workspaceSnapshot, fallbackProjects, t, list])
      const projectsAvailable = workspaceItems.length > 0
      const pickerWorkspaces = useMemo(() => ({ items: workspaceItems }), [workspaceItems])
      const selectedWorkspace = useMemo(
        () => workspaceItems.find(workspace => workspace.workspaceId === ui.workspaceId) || null,
        [workspaceItems, ui.workspaceId],
      )
      const selectedWorkspaceTitle = selectedWorkspace
        ? selectedWorkspace.title
        : ui.workspaceId ? t('project.noProject') : (workspaceItems[0] ? workspaceItems[0].title : t('project.noProject'))
      const projectsSettled = !workspaceSnapshot || workspaceSnapshot.phase === 'ready'
      const workspaceRef = useRef(selectedWorkspace)
      workspaceRef.current = selectedWorkspace
      const projectSignature = useMemo(() => {
        if (!selectedWorkspace) return ''
        const byId = list && list.byId ? list.byId : {}
        const chatSignature = (selectedWorkspace.sessionIds || []).map(id => {
          const row = byId[id]
          if (!row) return id + ':-'
          return [id, row.displayTitle || '', row.running ? 1 : 0, row.updatedAt || 0, row.parentId || '', row.blank ? 1 : 0].join('|')
        }).join(';')
        return [selectedWorkspace.workspaceId, selectedWorkspace.title, selectedWorkspace.path, chatSignature].join('::')
      }, [selectedWorkspace, list])

      const selectedRow = ui.sessionId && list && list.byId ? list.byId[ui.sessionId] : null
      const selectedTitle = snapshot.sessionId === ui.sessionId && snapshot.title
        ? snapshot.title
        : selectedRow ? selectedRow.displayTitle : ''

      const close = useCallback(() => {
        runtime.selectNode(null)
        uiStore.patch({ open: false, sector: null, mediaId: null, fileId: null })
      }, [runtime, uiStore])

      const onSelect = useCallback(id => {
        runtime.selectNode(id)
        if (id) uiStore.patch({ tab: 'node', mediaId: null, fileId: null })
      }, [runtime, uiStore])

      const onFocusNode = useCallback(id => {
        setFocusRequest({ id, at: Date.now() })
      }, [])

      const loadMedia = useCallback((sessionId, attachment) => {
        const service = ctx && ctx.uiConversation
        if (!service || typeof service.imageUrl !== 'function') {
          return Promise.reject(new Error('uiConversation imageUrl is unavailable'))
        }
        return service.imageUrl(sessionId, attachment)
      }, [ctx])

      const onOpenMedia = useCallback(media => {
        if (!media) return
        uiStore.patch({ tab: 'media', mediaId: media.id, fileId: null })
      }, [uiStore])

      const onOpenFile = useCallback(file => {
        if (!file) return
        uiStore.patch({ tab: 'files', fileId: file.id, mediaId: null })
      }, [uiStore])

      const onPickSession = useCallback(id => {
        runtime.selectNode(null)
        const workspace = workspaceItems.find(item =>
          Array.isArray(item.sessionIds) && item.sessionIds.includes(id))
        uiStore.patch({
          mode: 'chat',
          sessionId: id,
          workspaceId: workspace ? workspace.workspaceId : ui.workspaceId,
          tab: 'flow',
          sector: null,
          mediaId: null,
          fileId: null,
        })
      }, [runtime, uiStore, workspaceItems, ui.workspaceId])

      const onPickWorkspace = useCallback(id => {
        runtime.selectNode(null)
        uiStore.patch({ mode: 'project', workspaceId: id, tab: 'chats', sector: null, mediaId: null, fileId: null })
      }, [runtime, uiStore])

      const onOpenSession = useCallback(id => {
        runtime.selectNode(null)
        uiStore.patch({ mode: 'chat', sessionId: id, tab: 'flow', sector: null, filter: '', mediaId: null, fileId: null })
      }, [runtime, uiStore])

      const onModeChange = useCallback(mode => {
        if (mode === ui.mode) return
        runtime.selectNode(null)
        uiStore.patch({ mode, tab: mode === 'project' ? 'chats' : 'flow', sector: null, mediaId: null, fileId: null })
      }, [runtime, uiStore, ui.mode])

      const refresh = useCallback(() => {
        if (typeof ctx.sessions.refresh !== 'function') return
        setRefreshing(true)
        Promise.resolve(ctx.sessions.refresh()).catch(() => {}).then(() => setRefreshing(false))
      }, [ctx])

      useEffect(() => {
        if (!ui.open) return undefined
        previousFocusRef.current = document.activeElement
        if (consoleRef.current) consoleRef.current.focus()
        const timer = setTimeout(() => {
          if (consoleRef.current) consoleRef.current.focus()
        }, 30)
        return () => {
          clearTimeout(timer)
          const previous = previousFocusRef.current
          if (previous && typeof previous.focus === 'function') {
            try { previous.focus() } catch {}
          }
        }
      }, [ui.open])

      useEffect(() => {
        if (!ui.open) return undefined
        const onKeyDown = event => {
          if (event.key === 'Escape') {
            event.stopPropagation()
            close()
            return
          }
          if (event.key !== 'Tab') return
          const root = consoleRef.current
          if (!root) return
          const focusable = root.querySelectorAll(
            'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])',
          )
          if (!focusable.length) return
          const first = focusable[0]
          const last = focusable[focusable.length - 1]
          if (event.shiftKey && (document.activeElement === first || document.activeElement === root)) {
            event.preventDefault()
            last.focus()
          } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault()
            first.focus()
          }
        }
        document.addEventListener('keydown', onKeyDown)
        return () => document.removeEventListener('keydown', onKeyDown)
      }, [ui.open, close])

      useEffect(() => {
        if (!ui.open) return undefined
        const onWheel = event => {
          const target = event.target
          if (target && typeof target.closest === 'function' && target.closest('[data-nt-scroll]')) return
          event.preventDefault()
        }
        document.addEventListener('wheel', onWheel, { passive: false })
        return () => document.removeEventListener('wheel', onWheel)
      }, [ui.open])

      useEffect(() => {
        runtime.setActive(ui.open)
        return () => { runtime.setActive(false) }
      }, [ui.open, runtime])

      useEffect(() => {
        if (!ui.open) return undefined
        refresh()
        const timer = setInterval(() => { refresh() }, 30000)
        return () => clearInterval(timer)
      }, [ui.open, refresh])

      useEffect(() => {
        if (!ui.open || ui.mode !== 'project' || !projectsSettled || projectsAvailable) return
        uiStore.patch({ mode: 'chat', tab: 'flow' })
      }, [ui.open, ui.mode, projectsSettled, projectsAvailable, uiStore])

      useEffect(() => {
        if (!ui.open || ui.mode !== 'chat' || ui.sessionId) return
        const ids = list && Array.isArray(list.ids) ? list.ids : []
        const byId = list && list.byId ? list.byId : {}
        const runningId = ids.find(id => byId[id] && byId[id].running)
        const candidate = runningId || ids[0]
        if (candidate) uiStore.patch({ sessionId: candidate })
      }, [ui.open, ui.mode, ui.sessionId, list, uiStore])

      useEffect(() => {
        if (!ui.open || ui.mode !== 'project' || !ui.workspaceId) return
        if (selectedWorkspace) return
        const first = workspaceItems[0]
        if (first) uiStore.patch({ workspaceId: first.workspaceId })
      }, [ui.open, ui.mode, ui.workspaceId, selectedWorkspace, workspaceItems, uiStore])

      useEffect(() => {
        if (!ui.open || ui.mode !== 'project' || ui.workspaceId) return
        const currentWorkspace = workspaceItems.find(workspace =>
          Array.isArray(workspace.sessionIds) && workspace.sessionIds.includes(ui.sessionId))
        const runningWorkspace = workspaceItems.find(workspace => {
          const byId = list && list.byId ? list.byId : {}
          return Array.isArray(workspace.sessionIds) && workspace.sessionIds.some(id => byId[id] && byId[id].running)
        })
        const candidate = currentWorkspace || runningWorkspace || workspaceItems[0]
        if (candidate) uiStore.patch({ workspaceId: candidate.workspaceId })
      }, [ui.open, ui.mode, ui.workspaceId, ui.sessionId, workspaceItems, list, uiStore])

      useEffect(() => {
        if (!ui.open || ui.mode !== 'chat' || !ui.sessionId) return
        runtime.select(ui.sessionId)
      }, [ui.open, ui.mode, ui.sessionId, runtime])

      useEffect(() => {
        if (!ui.open || ui.mode !== 'project') return
        const workspace = workspaceRef.current
        if (!workspace) return
        runtime.showProject(workspace, ctx.sessions.list.getSnapshot())
      }, [ui.open, ui.mode, ui.workspaceId, projectSignature, runtime, ctx])

      if (!ui.open) return null

      const projectMode = ui.mode === 'project' && !!snapshot.project
      const modelText = projectMode
        ? t('project.chatCount', { count: snapshot.project ? snapshot.project.total : 0 })
        : snapshot.model && snapshot.model.model
          ? [snapshot.model.provider, snapshot.model.model].filter(Boolean).join(' · ')
          : t('model.none')
      const replaying = !!(snapshot.playback && snapshot.playback.active)
      const status = replaying
        ? t('timeline.replaying')
        : projectMode && snapshot.project && snapshot.project.running
          ? t('session.runningCount', { count: snapshot.project.running })
          : snapshot.running ? t('status.thinking') : t('status.idle')
      const openMedia = ui.mediaId && Array.isArray(snapshot.media)
        ? snapshot.media.find(item => item.id === ui.mediaId) || null
        : null
      const openFile = ui.fileId && Array.isArray(snapshot.files)
        ? snapshot.files.find(item => item.id === ui.fileId) || null
        : null
      const sideWidth = sideWidthValue(ui.sideWidth)

      return h('div', {
        className: 'dsh-nt-overlay' + (ui.fullscreen ? ' is-full' : '') + (IS_MAC ? ' is-mac' : ''),
        role: 'dialog',
        'aria-modal': 'true',
        'aria-label': t('panel.title'),
      },
        h('div', { className: 'dsh-nt-backdrop', onClick: close, 'aria-hidden': 'true' }),
        h('div', {
          className: 'dsh-nt-console' + (ui.fullscreen ? ' is-full' : ''),
          ref: consoleRef,
          tabIndex: -1,
        },
        h('header', { className: 'dsh-nt-head' },
          h('div', { className: 'dsh-nt-brand' },
            h('span', { className: 'dsh-nt-brand-mark' }, h(BrainMark, { size: 30 })),
            h('div', { className: 'dsh-nt-brand-text' },
              h('h2', null, t('panel.title')),
              h('span', null, t('panel.subtitle'))
            )
          ),
          h(ModeToggle, {
            mode: ui.mode,
            t,
            onChange: onModeChange,
            projectsAvailable,
          }),
          h(SessionPicker, {
            mode: ui.mode,
            list,
            selectedId: ui.sessionId,
            selectedTitle,
            workspaces: pickerWorkspaces,
            selectedWorkspaceId: ui.workspaceId,
            selectedWorkspaceTitle,
            t,
            locale,
            onPick: onPickSession,
            onPickWorkspace,
            onRefresh: refresh,
            refreshing,
          }),
          h('div', { className: 'dsh-nt-head-right' },
            h('span', { className: 'dsh-nt-model-chip', title: modelText }, modelText),
            h('span', { className: 'dsh-nt-live' + (snapshot.running ? ' is-running' : '') + (replaying ? ' is-history' : ''), role: 'status' },
              h('span', { className: 'dsh-nt-live-dot', 'aria-hidden': 'true' }),
              status
            ),
            h(LanguageMenu, { t }),
            h('button', {
              type: 'button',
              className: 'dsh-nt-icon-button',
              title: ui.fullscreen ? t('stage.exitFullscreen') : t('stage.fullscreen'),
              'aria-label': ui.fullscreen ? t('stage.exitFullscreen') : t('stage.fullscreen'),
              'aria-pressed': ui.fullscreen,
              onClick: () => uiStore.patch({ fullscreen: !ui.fullscreen }),
            }, h(IconFullscreen)),
            h('button', {
              type: 'button',
              className: 'dsh-nt-icon-button',
              title: t('panel.close'),
              'aria-label': t('panel.close'),
              onClick: close,
            }, h(IconClose))
          )
        ),
        h('div', { className: 'dsh-nt-body', style: { '--nt-side-width': sideWidth + 'px' } },
          h(BrainStage, {
            runtime, snapshot, ui, uiStore, t, locale,
            onSelect,
            onOpenSession,
            onOpenMedia,
            onOpenFile,
            loadMedia,
            focusRequest,
            listPending: !list || list.phase === 'pending' || list.phase === 'loading',
            onRetry: () => {
              const workspace = workspaceRef.current
              if (ui.mode === 'project' && workspace) {
                runtime.showProject(workspace, ctx.sessions.list.getSnapshot())
              } else {
                runtime.reload()
              }
            },
          }),
          h(SidePanel, {
            runtime, snapshot, ui, uiStore, t, locale,
            onSelect,
            onFocusNode,
            onOpenSession,
            onOpenMedia,
            onOpenFile,
            loadMedia,
          })
        )
        ),
        openMedia ? h(MediaViewer, {
          media: openMedia,
          sessionId: snapshot.sessionId,
          loadMedia,
          t,
          locale,
          onClose: () => uiStore.patch({ mediaId: null }),
          onSelectNode: onSelect,
        }) : null,
        openFile ? h(TextViewer, {
          file: openFile,
          t,
          locale,
          onClose: () => uiStore.patch({ fileId: null }),
          onSelectNode: onSelect,
        }) : null
      )
    }

    function NeuralPill(props) {
      const uiStore = props.uiStore
      const ui = useUi(uiStore)
      const t = useT()
      const sessionId = props.sessionId || (props.session && props.session.sessionId)
      const list = useSessionList(props.ctx)
      const row = sessionId && list && list.byId ? list.byId[sessionId] : null
      const running = !!(row && row.running)
      const title = t('pill.title')

      return h('button', {
        type: 'button',
        className: 'dsh-nt-pill' + (ui.open ? ' is-open' : ''),
        title,
        'aria-label': title,
        onClick: () => {
          props.runtime.selectNode(null)
          uiStore.patch({
            open: true,
            mode: 'chat',
            sessionId: sessionId || props.sessionId || null,
            tab: 'flow',
            sector: null,
          })
        },
      },
      h('span', { className: 'dsh-nt-pill-mark', 'aria-hidden': 'true' }, h(BrainMark, { size: 15 })),
      h('span', { className: 'dsh-nt-pill-label' }, t('pill.label')),
      running ? h('span', { className: 'dsh-nt-pill-dot', 'aria-hidden': 'true' }) : null
      )
    }

    /* ------------------------------------------------------------------ *
     *  Styles
     * ------------------------------------------------------------------ */

    const CSS = `
.dsh-nt-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 22px;
  padding: 0 9px 0 6px;
  margin: 0;
  border-radius: 999px;
  border: 1px solid var(--dsw-alias-border-l1);
  background: var(--dsw-alias-bg-layer-1);
  color: var(--dsw-alias-label-secondary);
  font: inherit;
  font-size: 10.5px;
  line-height: 1;
  letter-spacing: 0.01em;
  cursor: pointer;
  transition: color 140ms ease, border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
}
.dsh-nt-pill:hover {
  color: var(--dsw-alias-label-primary);
  border-color: var(--dsw-alias-brand-primary);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--dsw-alias-brand-primary) 24%, transparent), 0 4px 18px rgba(40, 170, 255, 0.16);
}
.dsh-nt-pill:active { transform: translateY(0.5px); }
.dsh-nt-pill.is-open { color: var(--dsw-alias-label-primary); border-color: var(--dsw-alias-brand-primary); }
.dsh-nt-pill-mark { display: inline-flex; color: var(--dsw-alias-brand-primary); }
.dsh-nt-pill-label { white-space: nowrap; }
.dsh-nt-pill-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: #47f0a8;
  box-shadow: 0 0 7px rgba(71, 240, 168, 0.9);
  animation: nt-pulse 1.7s ease-in-out infinite;
}

.dsh-nt-overlay {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 22px;
  pointer-events: none;
  font-family: inherit;
}
.dsh-nt-backdrop {
  position: absolute;
  inset: 0;
  pointer-events: auto;
  background:
    radial-gradient(1100px 640px at 50% 28%, rgba(16, 42, 78, 0.58), rgba(2, 4, 10, 0.9) 72%),
    rgba(2, 4, 10, 0.55);
  -webkit-backdrop-filter: blur(10px) saturate(115%);
  backdrop-filter: blur(10px) saturate(115%);
  animation: nt-fade 200ms ease-out both;
}
.dsh-nt-console {
  position: relative;
  pointer-events: auto;
  width: min(1280px, 96vw);
  height: min(860px, 90vh);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-radius: 18px;
  border: 1px solid rgba(104, 202, 255, 0.17);
  background:
    radial-gradient(900px 480px at 18% -10%, rgba(34, 96, 168, 0.24), transparent 62%),
    radial-gradient(760px 420px at 100% 0%, rgba(120, 70, 210, 0.16), transparent 58%),
    linear-gradient(158deg, #070c16 0%, #080e1a 45%, #050810 100%);
  box-shadow:
    0 44px 130px rgba(0, 0, 0, 0.72),
    0 0 0 1px rgba(150, 230, 255, 0.05) inset,
    0 0 110px rgba(38, 132, 255, 0.12);
  color: #eaf4ff;
  animation: nt-rise 240ms cubic-bezier(0.2, 0.9, 0.25, 1) both;
  outline: none;
}
.dsh-nt-console::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  opacity: 0.55;
  background-image:
    radial-gradient(1px 1px at 12% 18%, rgba(170, 230, 255, 0.75), transparent 60%),
    radial-gradient(1px 1px at 78% 12%, rgba(190, 160, 255, 0.6), transparent 60%),
    radial-gradient(1px 1px at 34% 82%, rgba(140, 240, 255, 0.55), transparent 60%),
    radial-gradient(1px 1px at 88% 66%, rgba(255, 160, 230, 0.45), transparent 60%),
    radial-gradient(1px 1px at 54% 34%, rgba(180, 240, 255, 0.55), transparent 60%),
    radial-gradient(1200px 500px at 50% 110%, rgba(30, 90, 170, 0.14), transparent 70%);
  background-size: 100% 100%;
}
.dsh-nt-console::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    repeating-linear-gradient(0deg, rgba(180, 230, 255, 0.022) 0 1px, transparent 1px 3px),
    radial-gradient(120% 90% at 50% 40%, transparent 55%, rgba(0, 0, 0, 0.45) 100%);
  mix-blend-mode: screen;
  opacity: 0.5;
}
.dsh-nt-head {
  position: relative;
  z-index: 3;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px 16px;
  flex: 0 0 auto;
  padding: 13px 16px;
  border-bottom: 1px solid rgba(112, 198, 255, 0.12);
  background: linear-gradient(180deg, rgba(13, 27, 48, 0.74), rgba(6, 11, 20, 0.32));
}
.dsh-nt-brand { display: flex; align-items: center; gap: 11px; min-width: 0; flex: 0 1 auto; }
.dsh-nt-brand-mark { display: inline-flex; filter: drop-shadow(0 0 10px rgba(90, 224, 255, 0.45)); }
.dsh-nt-brand-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.dsh-nt-brand-text h2 {
  margin: 0;
  font-size: 15px;
  font-weight: 620;
  letter-spacing: 0.015em;
  background: linear-gradient(96deg, #eafcff 8%, #8fe8ff 46%, #b79bff 92%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  white-space: nowrap;
}
.dsh-nt-brand-text span {
  font-size: 10px;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: rgba(170, 208, 245, 0.52);
  white-space: nowrap;
}
.dsh-nt-head-right { display: flex; align-items: center; gap: 10px; margin-left: auto; min-width: 0; }
.dsh-nt-model-chip {
  max-width: 230px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 5px 9px;
  border-radius: 8px;
  border: 1px solid rgba(120, 198, 255, 0.16);
  background: rgba(9, 19, 34, 0.7);
  color: rgba(200, 230, 255, 0.72);
  font-size: 10.5px;
  letter-spacing: 0.03em;
}
.dsh-nt-live {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 10.5px;
  letter-spacing: 0.02em;
  color: rgba(175, 214, 248, 0.62);
  white-space: nowrap;
}
.dsh-nt-live-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: rgba(140, 180, 220, 0.6);
}
.dsh-nt-live.is-running { color: #8df5c4; }
.dsh-nt-live.is-running .dsh-nt-live-dot {
  background: #47f0a8;
  box-shadow: 0 0 9px rgba(71, 240, 168, 0.95);
  animation: nt-pulse 1.5s ease-in-out infinite;
}
.dsh-nt-icon-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px; height: 28px;
  padding: 0;
  border-radius: 8px;
  border: 1px solid rgba(120, 200, 255, 0.16);
  background: rgba(10, 20, 36, 0.66);
  color: rgba(200, 232, 255, 0.78);
  cursor: pointer;
  transition: border-color 140ms ease, color 140ms ease, background 140ms ease, box-shadow 140ms ease;
}
.dsh-nt-icon-button:hover {
  color: #ecfaff;
  border-color: rgba(110, 220, 255, 0.5);
  background: rgba(18, 38, 64, 0.9);
  box-shadow: 0 0 16px rgba(64, 180, 255, 0.18);
}
.dsh-nt-icon-button:focus-visible, .dsh-nt-pill:focus-visible,
.dsh-nt-ghost:focus-visible, .dsh-nt-primary:focus-visible,
.dsh-nt-tab:focus-visible, .dsh-nt-legend-chip:focus-visible,
.dsh-nt-session-row:focus-visible, .dsh-nt-toast:focus-visible,
.dsh-nt-signal:focus-visible, .dsh-nt-flow-item:focus-visible,
.dsh-nt-link-chip:focus-visible {
  outline: 1px solid rgba(130, 225, 255, 0.9);
  outline-offset: 2px;
}

.dsh-nt-picker { position: relative; z-index: 4; flex: 1 1 240px; min-width: 180px; max-width: 380px; }
.dsh-nt-picker-button {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-width: 180px;
  max-width: 380px;
  padding: 6px 10px;
  border-radius: 10px;
  border: 1px solid rgba(120, 200, 255, 0.16);
  background: rgba(9, 18, 32, 0.78);
  color: #eaf4ff;
  text-align: left;
  cursor: pointer;
  transition: border-color 140ms ease, box-shadow 140ms ease, background 140ms ease;
}
.dsh-nt-picker-button:hover {
  border-color: rgba(110, 220, 255, 0.42);
  background: rgba(12, 26, 46, 0.92);
}
.dsh-nt-picker-text { display: flex; flex-direction: column; gap: 1px; min-width: 0; flex: 1 1 auto; }
.dsh-nt-picker-eyebrow {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(172, 214, 250, 0.64);
}
.dsh-nt-picker-running {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 1px 5px;
  border-radius: 999px;
  border: 1px solid rgba(71, 240, 168, 0.28);
  background: rgba(24, 92, 68, 0.22);
  color: #8df5c4;
  font-size: 8px;
  letter-spacing: 0.06em;
}
.dsh-nt-picker-running-dot {
  width: 4px; height: 4px; border-radius: 50%;
  background: #47f0a8;
  box-shadow: 0 0 6px rgba(71, 240, 168, 0.9);
  animation: nt-pulse 1.5s ease-in-out infinite;
}
.dsh-nt-picker-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12.5px;
  font-weight: 560;
}
.dsh-nt-picker-chevron { display: inline-flex; color: rgba(170, 215, 250, 0.66); flex: 0 0 auto; }
.dsh-nt-picker-pop {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  width: min(420px, 56vw);
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  border-radius: 14px;
  border: 1px solid rgba(110, 205, 255, 0.26);
  background: linear-gradient(180deg, rgba(10, 20, 36, 0.98), rgba(6, 11, 20, 0.99));
  box-shadow: 0 26px 70px rgba(0, 0, 0, 0.7), 0 0 40px rgba(40, 140, 255, 0.1);
  z-index: 20;
  animation: nt-pop 150ms ease-out both;
}
.dsh-nt-picker-search {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 6px 9px;
  border-radius: 9px;
  border: 1px solid rgba(120, 200, 255, 0.16);
  background: rgba(5, 10, 18, 0.8);
  color: rgba(160, 205, 245, 0.62);
}
.dsh-nt-picker-search input {
  flex: 1 1 auto;
  min-width: 0;
  border: 0;
  background: transparent;
  color: #eaf4ff;
  font: inherit;
  font-size: 12px;
  outline: none;
}
.dsh-nt-picker-search input::placeholder { color: rgba(150, 195, 235, 0.4); }
.dsh-nt-session-list { max-height: 306px; overflow: auto; overscroll-behavior: contain; display: flex; flex-direction: column; gap: 2px; }
.dsh-nt-session-row {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 7px 8px;
  border: 1px solid transparent;
  border-radius: 9px;
  background: transparent;
  color: #eaf4ff;
  text-align: left;
  cursor: pointer;
}
.dsh-nt-session-row:hover { background: rgba(26, 52, 84, 0.5); }
.dsh-nt-session-row.is-current { border-color: rgba(110, 220, 255, 0.34); background: rgba(20, 44, 74, 0.6); }
.dsh-nt-session-state {
  width: 7px; height: 7px; border-radius: 50%;
  background: rgba(130, 170, 210, 0.42);
  flex: 0 0 auto;
}
.dsh-nt-session-state.is-running {
  background: #47f0a8;
  box-shadow: 0 0 8px rgba(71, 240, 168, 0.9);
  animation: nt-pulse 1.5s ease-in-out infinite;
}
.dsh-nt-session-main { display: flex; flex-direction: column; gap: 1px; min-width: 0; flex: 1 1 auto; }
.dsh-nt-session-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; font-weight: 540; }
.dsh-nt-session-meta {
  display: flex; gap: 8px; align-items: baseline;
  font-size: 10px; color: rgba(177, 214, 246, 0.64);
  overflow: hidden; white-space: nowrap;
}
.dsh-nt-session-cwd { overflow: hidden; text-overflow: ellipsis; opacity: 0.72; }
.dsh-nt-session-check { display: inline-flex; color: #6fe9ff; flex: 0 0 auto; }
.dsh-nt-session-empty { display: flex; flex-direction: column; gap: 6px; padding: 18px 12px; color: rgba(180, 215, 245, 0.6); font-size: 11.5px; }
.dsh-nt-session-empty strong { color: #eaf4ff; font-size: 12.5px; }
.dsh-nt-picker-foot {
  display: flex; align-items: center; justify-content: space-between;
  padding: 4px 4px 0;
  border-top: 1px solid rgba(120, 200, 255, 0.1);
  font-size: 10px; color: rgba(174, 212, 244, 0.64);
}
.dsh-nt-ghost {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 9px;
  border-radius: 8px;
  border: 1px solid rgba(120, 200, 255, 0.18);
  background: rgba(12, 24, 40, 0.6);
  color: rgba(205, 235, 255, 0.82);
  font: inherit; font-size: 11px;
  cursor: pointer;
}
.dsh-nt-ghost:hover { border-color: rgba(110, 220, 255, 0.46); color: #f0fbff; }
.dsh-nt-ghost:disabled { opacity: 0.55; cursor: default; }
.dsh-nt-primary {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 12px;
  border-radius: 8px;
  border: 1px solid rgba(110, 220, 255, 0.45);
  background: linear-gradient(180deg, rgba(30, 108, 168, 0.62), rgba(16, 62, 104, 0.7));
  color: #eafcff;
  font: inherit; font-size: 11.5px; font-weight: 560;
  cursor: pointer;
  box-shadow: 0 0 22px rgba(52, 170, 255, 0.16);
}
.dsh-nt-primary:hover { border-color: rgba(140, 240, 255, 0.8); box-shadow: 0 0 28px rgba(60, 190, 255, 0.28); }

.dsh-nt-body {
  position: relative; z-index: 1; flex: 1 1 auto; min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) var(--nt-side-width, 440px);
}
.dsh-nt-stage {
  position: relative;
  min-width: 0;
  overflow: hidden;
  background:
    radial-gradient(880px 560px at 50% 46%, rgba(18, 52, 98, 0.32), rgba(3, 6, 13, 0) 70%),
    radial-gradient(420px 380px at 50% 50%, rgba(10, 26, 48, 0.25), transparent 70%);
}
.dsh-nt-canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
.dsh-nt-canvas:focus-visible { outline: 1px solid rgba(120, 222, 255, 0.72); outline-offset: -3px; }
.dsh-nt-sr-only {
  position: absolute;
  width: 1px; height: 1px;
  padding: 0; margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.dsh-nt-hud-top {
  position: absolute; top: 12px; left: 14px; right: 14px;
  display: flex; align-items: flex-start; justify-content: space-between; gap: 10px;
  pointer-events: none;
}
.dsh-nt-hud-title { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.dsh-nt-hud-kicker {
  font-size: 9px; letter-spacing: 0.24em; color: rgba(125, 226, 255, 0.64);
  text-shadow: 0 0 12px rgba(80, 200, 255, 0.4);
}
.dsh-nt-hud-counts { font-size: 11px; color: rgba(206, 233, 252, 0.78); letter-spacing: 0.02em; }
.dsh-nt-hud-calls { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 1px; pointer-events: auto; }
.dsh-nt-hud-call {
  display: inline-flex; align-items: center; gap: 4px;
  height: 19px; padding: 0 6px;
  border: 1px solid color-mix(in srgb, var(--nt-color) 38%, transparent);
  border-radius: 999px;
  background: color-mix(in srgb, var(--nt-color) 9%, rgba(5, 12, 22, 0.72));
  color: var(--nt-color);
  font: inherit; font-size: 8.5px; letter-spacing: 0.04em;
  cursor: pointer;
}
.dsh-nt-hud-call:hover { background: color-mix(in srgb, var(--nt-color) 17%, rgba(5, 12, 22, 0.8)); }
.dsh-nt-hud-call svg { width: 10px; height: 10px; }
.dsh-nt-hud-call-count {
  min-width: 12px; height: 12px; padding: 0 3px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--nt-color) 22%, transparent);
  color: #f3fbff; font-size: 8px; line-height: 12px; text-align: center;
}
.dsh-nt-hud-hint {
  font-size: 10px; color: rgba(255, 205, 132, 0.78); letter-spacing: 0.02em;
  max-width: 290px;
}
.dsh-nt-hud-cap {
  font-size: 10px; color: rgba(170, 214, 248, 0.66); letter-spacing: 0.02em;
}
.dsh-nt-hud-actions { display: flex; align-items: center; gap: 5px; pointer-events: auto; }

.dsh-nt-legend {
  position: absolute; left: 14px; bottom: 13px; right: 14px;
  display: flex; flex-wrap: wrap; align-items: flex-end; align-content: flex-end;
  gap: 5px;
  pointer-events: none;
}
.dsh-nt-legend-chip {
  pointer-events: auto;
  display: inline-flex; align-items: center; gap: 5px;
  flex: 0 0 auto; max-width: 100%;
  height: 23px; padding: 0 8px;
  border-radius: 999px;
  border: 1px solid rgba(120, 190, 245, 0.14);
  background: rgba(6, 13, 24, 0.66);
  color: rgba(203, 231, 252, 0.84);
  font: inherit; font-size: 10px; letter-spacing: 0.02em;
  white-space: nowrap; overflow: hidden;
  cursor: pointer;
  transition: border-color 140ms ease, color 140ms ease, background 140ms ease;
  -webkit-backdrop-filter: blur(6px);
  backdrop-filter: blur(6px);
}
.dsh-nt-legend-chip:hover { color: #eefaff; border-color: var(--nt-color); }
.dsh-nt-legend-chip.is-active {
  color: #f2fcff;
  border-color: var(--nt-color);
  background: color-mix(in srgb, var(--nt-color) 14%, rgba(6, 13, 24, 0.8));
  box-shadow: 0 0 16px color-mix(in srgb, var(--nt-color) 32%, transparent);
}
.dsh-nt-legend-dot {
  flex: 0 0 auto;
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--nt-color);
  box-shadow: 0 0 8px var(--nt-color);
}
.dsh-nt-legend-label { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dsh-nt-legend-count { flex: 0 0 auto; color: rgba(176, 214, 245, 0.66); font-variant-numeric: tabular-nums; }
/* ---- History timeline ---- */
.dsh-nt-timeline {
  position: absolute;
  left: 50%;
  bottom: 74px;
  transform: translateX(-50%);
  z-index: 6;
  width: min(760px, calc(100% - 32px));
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 9px;
  border-radius: 13px;
  border: 1px solid rgba(112, 198, 255, 0.18);
  background: linear-gradient(110deg, rgba(8, 18, 32, 0.94), rgba(5, 12, 23, 0.9));
  box-shadow: 0 14px 38px rgba(0, 0, 0, 0.45);
  backdrop-filter: blur(14px);
  pointer-events: auto;
}
.dsh-nt-timeline.is-active {
  border-color: rgba(111, 233, 255, 0.4);
  box-shadow: 0 14px 44px rgba(0, 0, 0, 0.55), 0 0 28px rgba(60, 190, 255, 0.12);
}
.dsh-nt-tl-controls { display: flex; align-items: center; gap: 3px; flex: 0 0 auto; }
.dsh-nt-tl-play {
  width: 30px; height: 30px;
  display: inline-flex; align-items: center; justify-content: center;
  border-radius: 50%;
  border: 1px solid rgba(111, 233, 255, 0.5);
  background: rgba(24, 86, 114, 0.55);
  color: #eafcff;
  cursor: pointer;
}
.dsh-nt-tl-play:hover { background: rgba(36, 122, 158, 0.7); border-color: #8ff1ff; }
.dsh-nt-tl-play:disabled { opacity: 0.35; cursor: default; }
.dsh-nt-tl-button {
  width: 25px; height: 25px;
  display: inline-flex; align-items: center; justify-content: center;
  border: 1px solid transparent;
  border-radius: 7px;
  background: transparent;
  color: rgba(183, 225, 250, 0.76);
  cursor: pointer;
}
.dsh-nt-tl-button:hover { color: #f2fdff; background: rgba(34, 78, 112, 0.42); }
.dsh-nt-tl-button.is-on { color: #8df5ff; border-color: rgba(111, 233, 255, 0.35); background: rgba(26, 70, 98, 0.5); }
.dsh-nt-tl-button:disabled { opacity: 0.28; cursor: default; }
.dsh-nt-tl-main { display: flex; flex-direction: column; gap: 1px; flex: 1 1 auto; min-width: 0; }
.dsh-nt-tl-meta { display: flex; align-items: center; gap: 8px; min-width: 0; font-size: 9px; color: rgba(171, 213, 244, 0.64); }
.dsh-nt-tl-badge { display: inline-flex; align-items: center; gap: 4px; letter-spacing: 0.05em; text-transform: uppercase; color: #9fe8ff; white-space: nowrap; }
.dsh-nt-tl-time { margin-left: auto; color: rgba(218, 242, 255, 0.85); font-variant-numeric: tabular-nums; white-space: nowrap; }
.dsh-nt-tl-count { color: rgba(140, 225, 250, 0.8); font-variant-numeric: tabular-nums; white-space: nowrap; }
.dsh-nt-tl-track { position: relative; display: flex; align-items: center; height: 17px; }
.dsh-nt-tl-track input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 16px;
  margin: 0;
  background: transparent;
  cursor: pointer;
}
.dsh-nt-tl-track input[type="range"]:disabled { cursor: default; opacity: 0.45; }
.dsh-nt-tl-track input[type="range"]::-webkit-slider-runnable-track {
  height: 3px;
  border-radius: 999px;
  background: rgba(111, 170, 210, 0.26);
}
.dsh-nt-timeline.is-active .dsh-nt-tl-track input[type="range"]::-webkit-slider-runnable-track {
  background: linear-gradient(90deg, rgba(111, 233, 255, 0.75), rgba(111, 233, 255, 0.22));
}
.dsh-nt-tl-track input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 11px; height: 11px;
  margin-top: -4px;
  border-radius: 50%;
  border: 2px solid #061421;
  background: #9ff3ff;
  box-shadow: 0 0 8px rgba(111, 233, 255, 0.8);
}
.dsh-nt-timeline:not(.is-active) .dsh-nt-tl-track input[type="range"]::-webkit-slider-thumb { background: #6c8ca6; box-shadow: none; }
.dsh-nt-tl-markers { position: absolute; left: 5px; right: 5px; top: 50%; height: 0; pointer-events: none; }
.dsh-nt-tl-marker {
  position: absolute;
  top: -4px;
  width: 6px; height: 6px;
  margin-left: -3px;
  padding: 0;
  border-radius: 50%;
  border: 1px solid rgba(4, 12, 20, 0.8);
  opacity: 0.75;
  cursor: pointer;
  pointer-events: auto;
  transform: scale(0.85);
  transition: transform 120ms ease, opacity 120ms ease;
}
.dsh-nt-tl-marker:hover, .dsh-nt-tl-marker:focus-visible { opacity: 1; transform: scale(1.35); outline: none; }
.dsh-nt-tl-extra { display: flex; align-items: center; gap: 4px; flex: 0 0 auto; }
.dsh-nt-tl-speed select {
  height: 25px;
  padding: 0 4px;
  border-radius: 7px;
  border: 1px solid rgba(112, 198, 255, 0.16);
  background: rgba(10, 26, 44, 0.8);
  color: #cfeeff;
  font: inherit;
  font-size: 9.5px;
  cursor: pointer;
}
.dsh-nt-tl-live {
  display: inline-flex; align-items: center; gap: 4px;
  height: 25px; padding: 0 7px;
  border-radius: 7px;
  border: 1px solid rgba(111, 233, 255, 0.35);
  background: rgba(20, 65, 88, 0.5);
  color: #bdf4ff;
  font: inherit; font-size: 9.5px;
  cursor: pointer;
}
.dsh-nt-tl-live:hover { background: rgba(32, 98, 126, 0.65); }
.dsh-nt-live.is-history { color: #9fe8ff; }
.dsh-nt-live.is-history .dsh-nt-live-dot { background: #9fe8ff; box-shadow: 0 0 10px #9fe8ff; }

@media (max-width: 900px) {
  .dsh-nt-timeline { bottom: 66px; flex-wrap: wrap; }
  .dsh-nt-tl-main { flex-basis: calc(100% - 44px); }
  .dsh-nt-tl-extra { margin-left: auto; }
}

.dsh-nt-stage-hint {
  position: absolute; left: 50%; bottom: 46px;
  transform: translateX(-50%);
  font-size: 10px; letter-spacing: 0.04em;
  color: rgba(176, 214, 246, 0.56);
  pointer-events: none;
  white-space: nowrap;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
}
.dsh-nt-stage-overlay {
  position: absolute; inset: 0;
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 9px;
  padding: 24px;
  text-align: center;
  color: rgba(205, 232, 255, 0.7);
  font-size: 12px;
  background: radial-gradient(500px 320px at 50% 50%, rgba(4, 9, 17, 0.6), transparent 75%);
  pointer-events: none;
}
.dsh-nt-stage-overlay strong { color: #eaf6ff; font-size: 14px; font-weight: 600; }
.dsh-nt-stage-overlay span { max-width: 360px; line-height: 1.5; }
.dsh-nt-stage-overlay .dsh-nt-ghost,
.dsh-nt-stage-overlay button { pointer-events: auto; }
.dsh-nt-stage-overlay.is-error strong { color: #ffb2a8; }
.dsh-nt-stage-overlay.is-error { pointer-events: auto; }

.dsh-nt-toasts {
  position: absolute; top: 58px; left: 50%;
  transform: translateX(-50%);
  display: flex; flex-direction: column; gap: 6px;
  width: min(470px, calc(100% - 32px));
  pointer-events: none;
  z-index: 4;
}
.dsh-nt-live-thought {
  pointer-events: auto;
  display: flex;
  flex-direction: column;
  gap: 5px;
  width: 100%;
  padding: 9px 11px;
  border-radius: 12px;
  border: 1px solid color-mix(in srgb, var(--nt-color) 55%, transparent);
  background: linear-gradient(100deg, rgba(10, 22, 38, 0.96), rgba(7, 16, 28, 0.94));
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.55), 0 0 34px color-mix(in srgb, var(--nt-color) 22%, transparent);
  color: #eefaff;
  font: inherit;
  text-align: left;
  cursor: pointer;
  animation: nt-toast-in 220ms cubic-bezier(0.2, 0.9, 0.25, 1) both;
}
.dsh-nt-live-thought:hover { border-color: var(--nt-color); }
.dsh-nt-live-thought-head { display: flex; align-items: center; gap: 7px; }
.dsh-nt-live-thought-icon { display: inline-flex; color: var(--nt-color); animation: nt-pulse 2s ease-in-out infinite; }
.dsh-nt-live-thought-kind { font-size: 8.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--nt-color); }
.dsh-nt-live-thought-sector { font-size: 8.5px; letter-spacing: 0.04em; color: rgba(166, 212, 246, 0.58); }
.dsh-nt-live-thought-badge {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 8.5px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: #8df5c4;
}
.dsh-nt-live-thought-text {
  font-size: 11.5px;
  line-height: 1.55;
  color: rgba(226, 242, 255, 0.9);
  white-space: pre-wrap;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.dsh-nt-toast {
  pointer-events: auto;
  display: flex; align-items: center; gap: 9px;
  width: 100%;
  padding: 8px 10px;
  border-radius: 11px;
  border: 1px solid color-mix(in srgb, var(--nt-color) 46%, transparent);
  background: linear-gradient(100deg, rgba(7, 16, 28, 0.94), rgba(10, 22, 38, 0.9));
  box-shadow: 0 12px 34px rgba(0, 0, 0, 0.52), 0 0 26px color-mix(in srgb, var(--nt-color) 22%, transparent);
  color: #eefaff;
  font: inherit;
  text-align: left;
  cursor: pointer;
  animation: nt-toast-in 240ms cubic-bezier(0.2, 0.9, 0.25, 1) both;
}
.dsh-nt-toast:hover { border-color: var(--nt-color); }
.dsh-nt-toast.is-call {
  border-color: color-mix(in srgb, var(--nt-color) 34%, transparent);
  box-shadow: 0 0 24px color-mix(in srgb, var(--nt-color) 12%, transparent), 0 18px 44px rgba(0, 0, 0, 0.52);
}
.dsh-nt-toast.is-call .dsh-nt-toast-icon {
  border-color: color-mix(in srgb, var(--nt-color) 38%, transparent);
  color: var(--nt-color);
  background: color-mix(in srgb, var(--nt-color) 11%, transparent);
}
.dsh-nt-toast.is-call .dsh-nt-toast-sector { color: var(--nt-color); opacity: 0.92; }
.dsh-nt-toast-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 7px;
  color: var(--nt-color);
  background: color-mix(in srgb, var(--nt-color) 16%, transparent);
  flex: 0 0 auto;
}
.dsh-nt-toast-body { display: flex; flex-direction: column; gap: 1px; min-width: 0; flex: 1 1 auto; }
.dsh-nt-toast-name { font-size: 11.5px; font-weight: 580; letter-spacing: 0.01em; }
.dsh-nt-toast-summary {
  font-size: 10px; color: rgba(195, 226, 249, 0.75);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
".dsh-nt-toast-sector {
  flex: 0 0 auto;
  font-size: 9px; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--nt-color);
  padding: 3px 6px;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--nt-color) 38%, transparent);
}
.dsh-nt-toast.is-media { padding: 7px 9px; }
.dsh-nt-toast.is-media .dsh-nt-toast-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dsh-nt-media-toast-thumb {
  flex: 0 0 auto;
  width: 58px; height: 42px;
  display: flex; align-items: center; justify-content: center;
  overflow: hidden;
  border-radius: 8px;
  border: 1px solid color-mix(in srgb, var(--nt-color) 32%, transparent);
  background: #02060c;
}
.dsh-nt-media-toast-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }

/* ---- Media tab ---- */
.dsh-nt-media { display: flex; flex-direction: column; gap: 10px; }
.dsh-nt-media-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 6px; }
.dsh-nt-media-filters { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
.dsh-nt-media-filter {
  display: inline-flex; align-items: center; gap: 4px;
  height: 24px; padding: 0 7px;
  border-radius: 7px;
  border: 1px solid rgba(112, 198, 255, 0.14);
  background: rgba(12, 25, 43, 0.5);
  color: rgba(182, 219, 247, 0.7);
  font: inherit; font-size: 9.5px;
  cursor: pointer;
}
.dsh-nt-media-filter:hover { color: #eaf9ff; border-color: rgba(112, 198, 255, 0.34); }
.dsh-nt-media-filter.is-active { color: #eafcff; border-color: rgba(111, 233, 255, 0.5); background: rgba(25, 66, 96, 0.55); }
.dsh-nt-media-filter-count {
  min-width: 14px; height: 14px; padding: 0 4px;
  border-radius: 999px;
  background: rgba(110, 220, 255, 0.14);
  color: #a9e9ff; font-size: 8.5px; line-height: 14px; text-align: center;
}
.dsh-nt-media-dropped { font-size: 9px; color: rgba(255, 190, 120, 0.8); }
.dsh-nt-media-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.dsh-nt-media-tile {
  display: flex; flex-direction: column;
  min-width: 0; padding: 0;
  border: 1px solid rgba(112, 198, 255, 0.13);
  border-radius: 10px;
  overflow: hidden;
  background: rgba(8, 17, 30, 0.62);
  color: #eaf6ff; font: inherit; text-align: left;
  cursor: pointer;
}
.dsh-nt-media-tile:hover, .dsh-nt-media-tile.is-active { border-color: rgba(111, 233, 255, 0.5); box-shadow: 0 0 20px rgba(80, 190, 255, 0.12); }
.dsh-nt-media-tile-frame {
  position: relative;
  display: block;
  aspect-ratio: 16 / 10;
  background: radial-gradient(circle at 50% 40%, #101f32, #03070e);
  overflow: hidden;
}
.dsh-nt-media-tile-frame img { width: 100%; height: 100%; object-fit: cover; display: block; }
.dsh-nt-media-tile-body { display: flex; flex-direction: column; gap: 2px; padding: 6px 7px 7px; min-width: 0; }
.dsh-nt-media-tile-name { font-size: 10px; font-weight: 560; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dsh-nt-media-tile-meta { font-size: 8.5px; color: rgba(174, 211, 241, 0.62); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dsh-nt-media-placeholder { display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; color: rgba(150, 210, 245, 0.5); }
.dsh-nt-media-placeholder.is-error { color: #ff857c; }
.dsh-nt-media-play {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  color: rgba(238, 252, 255, 0.9);
  background: rgba(3, 8, 14, 0.28);
  pointer-events: none;
}
.dsh-nt-media-more {
  align-self: center;
  padding: 6px 11px;
  border-radius: 8px;
  border: 1px solid rgba(112, 198, 255, 0.2);
  background: rgba(15, 32, 54, 0.6);
  color: #b9e9ff; font: inherit; font-size: 9.5px; cursor: pointer;
}
.dsh-nt-media-empty { display: flex; flex-direction: column; align-items: center; gap: 7px; padding: 42px 16px; text-align: center; color: rgba(170, 210, 245, 0.6); }
.dsh-nt-media-empty-icon { color: rgba(110, 205, 255, 0.55); }
.dsh-nt-media-empty-title { font-size: 12px; color: #dff6ff; }
.dsh-nt-media-empty-hint { font-size: 10px; line-height: 1.5; max-width: 220px; }

/* ---- Files tab ---- */
.dsh-nt-files { display: flex; flex-direction: column; gap: 8px; }
.dsh-nt-file-list { display: flex; flex-direction: column; gap: 5px; }
.dsh-nt-file-row {
  display: flex; align-items: center; gap: 8px;
  width: 100%; padding: 7px 8px;
  border: 1px solid rgba(112, 198, 255, 0.12);
  border-radius: 9px;
  background: rgba(8, 17, 30, 0.58);
  color: #eaf6ff; font: inherit; text-align: left; cursor: pointer;
}
.dsh-nt-file-row:hover, .dsh-nt-file-row.is-active { border-color: rgba(111, 233, 255, 0.46); background: rgba(15, 36, 58, 0.68); }
.dsh-nt-file-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 28px; height: 28px; flex: 0 0 auto;
  border-radius: 7px;
  color: #9fdcff;
  background: rgba(80, 160, 240, 0.14);
}
.dsh-nt-file-body { display: flex; flex-direction: column; gap: 2px; min-width: 0; flex: 1 1 auto; }
.dsh-nt-file-name { font-size: 10.5px; font-weight: 560; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dsh-nt-file-meta { font-size: 8.5px; color: rgba(174, 211, 241, 0.62); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dsh-nt-file-desc { font-size: 9px; color: rgba(186, 219, 245, 0.5); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dsh-nt-file-arrow { display: inline-flex; color: rgba(130, 210, 250, 0.5); flex: 0 0 auto; }
.dsh-nt-file-toast-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 34px; height: 34px; flex: 0 0 auto;
  border-radius: 8px;
  color: #9fdcff;
  background: rgba(80, 160, 240, 0.16);
}
.dsh-nt-node-files { display: flex; flex-direction: column; gap: 4px; }
.dsh-nt-node-file {
  display: flex; align-items: center; gap: 6px;
  width: 100%; padding: 5px 7px;
  border: 1px solid rgba(112, 198, 255, 0.12);
  border-radius: 7px;
  background: rgba(6, 14, 25, 0.6);
  color: #dceeff; font: inherit; text-align: left; cursor: pointer;
}
.dsh-nt-node-file:hover { border-color: rgba(111, 233, 255, 0.44); }
.dsh-nt-node-file-name { flex: 1 1 auto; min-width: 0; font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dsh-nt-node-file-ext { flex: 0 0 auto; font-size: 8px; letter-spacing: 0.06em; color: rgba(160, 210, 245, 0.6); }

/* ---- Text viewer ---- */
.dsh-nt-text-viewer .dsh-nt-media-viewer-card { width: min(960px, 94vw); }
.dsh-nt-text-toolbar { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.dsh-nt-text-toolbar-spacer { flex: 1 1 auto; }
.dsh-nt-text-modes { display: flex; gap: 4px; }
.dsh-nt-text-toolbar .dsh-nt-ghost { text-decoration: none; }
.dsh-nt-text-stage {
  display: block;
  align-items: stretch;
  flex: 1 1 auto;
  min-height: 0;
  max-height: 60vh;
  overflow: auto;
  overscroll-behavior: contain;
  padding: 12px 14px;
  background: #02060c;
}
.dsh-nt-text-pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  color: rgba(221, 240, 255, 0.9);
  font: 11px/1.6 ui-monospace, "SF Mono", SFMono-Regular, Menlo, monospace;
  tab-size: 2;
}
.dsh-nt-text-note { font-size: 9.5px; color: rgba(255, 190, 120, 0.8); }
.dsh-nt-md { max-width: 780px; margin: 0 auto; color: #dceeff; font-size: 12px; line-height: 1.65; word-break: break-word; }
.dsh-nt-md h1, .dsh-nt-md h2, .dsh-nt-md h3, .dsh-nt-md h4, .dsh-nt-md h5, .dsh-nt-md h6 {
  margin: 1.1em 0 0.45em; color: #f2fbff; line-height: 1.25; font-weight: 650;
}
.dsh-nt-md h1 { font-size: 20px; }
.dsh-nt-md h2 { font-size: 17px; }
.dsh-nt-md h3 { font-size: 14px; }
.dsh-nt-md h4, .dsh-nt-md h5, .dsh-nt-md h6 { font-size: 12px; }
.dsh-nt-md p { margin: 0.55em 0; }
.dsh-nt-md a { color: #7fe8ff; text-decoration: underline; text-underline-offset: 2px; }
.dsh-nt-md code { padding: 1px 4px; border-radius: 4px; background: rgba(80, 160, 240, 0.14); color: #b7ecff; font: 10.5px/1.5 ui-monospace, Menlo, monospace; }
.dsh-nt-md pre { margin: 0.7em 0; padding: 9px 11px; overflow: auto; border-radius: 8px; border: 1px solid rgba(112, 198, 255, 0.12); background: rgba(2, 7, 14, 0.8); }
.dsh-nt-md pre code { padding: 0; background: none; color: #dceeff; font-size: 10.5px; }
.dsh-nt-md ul, .dsh-nt-md ol { margin: 0.5em 0; padding-left: 1.5em; }
.dsh-nt-md li { margin: 0.2em 0; }
.dsh-nt-md blockquote { margin: 0.7em 0; padding: 4px 10px; border-left: 2px solid rgba(111, 233, 255, 0.5); color: rgba(204, 231, 250, 0.75); }
.dsh-nt-md hr { border: 0; border-top: 1px solid rgba(112, 198, 255, 0.18); margin: 1em 0; }
.dsh-nt-md-table-wrap, .dsh-nt-csv-wrap { max-width: 100%; overflow: auto; margin: 0.7em 0; }
.dsh-nt-md-table, .dsh-nt-csv-table { border-collapse: collapse; width: 100%; font-size: 10.5px; }
.dsh-nt-md-table th, .dsh-nt-md-table td, .dsh-nt-csv-table th, .dsh-nt-csv-table td {
  padding: 5px 7px; border: 1px solid rgba(112, 198, 255, 0.14); text-align: left; vertical-align: top;
}
.dsh-nt-md-table th, .dsh-nt-csv-table th { background: rgba(30, 66, 100, 0.45); color: #eaf9ff; font-weight: 600; position: sticky; top: 0; }
.dsh-nt-csv-table td { color: rgba(220, 240, 255, 0.86); white-space: pre-wrap; }

/* ---- Media lightbox ---- */
.dsh-nt-media-viewer {
  position: fixed; inset: 0; z-index: 2000;
  display: flex; align-items: center; justify-content: center;
  padding: 24px;
  pointer-events: auto;
}
.dsh-nt-media-viewer-backdrop {
  position: absolute; inset: 0;
  border: 0;
  background: #02060c;
  cursor: pointer;
}
.dsh-nt-media-viewer-card {
  position: relative; z-index: 1;
  display: flex; flex-direction: column; gap: 10px;
  width: min(900px, 92vw);
  max-height: 90vh;
  overflow: auto;
  padding: 13px;
  border-radius: 15px;
  border: 1px solid rgba(112, 210, 255, 0.26);
  background: linear-gradient(180deg, rgba(9, 20, 35, 0.99), rgba(4, 9, 17, 0.99));
  box-shadow: 0 30px 100px rgba(0, 0, 0, 0.8), 0 0 60px rgba(70, 190, 255, 0.12);
  scrollbar-width: thin;
}
.dsh-nt-media-viewer-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.dsh-nt-media-viewer-title { font-size: 12px; font-weight: 600; letter-spacing: 0.01em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dsh-nt-media-viewer-stage {
  display: flex; align-items: center; justify-content: center;
  min-height: 200px; max-height: 56vh;
  overflow: hidden;
  border-radius: 10px;
  background: #02060c;
}
.dsh-nt-media-viewer-image, .dsh-nt-media-viewer-video { max-width: 100%; max-height: 56vh; object-fit: contain; display: block; }
.dsh-nt-media-viewer-status { display: inline-flex; align-items: center; gap: 7px; padding: 26px; color: rgba(175, 215, 245, 0.65); font-size: 11px; }
.dsh-nt-media-viewer-status.is-error { color: #ff857c; }
.dsh-nt-media-viewer-meta { display: flex; flex-direction: column; gap: 3px; }
.dsh-nt-media-meta-row { display: flex; justify-content: space-between; gap: 12px; font-size: 10px; padding: 3px 0; border-bottom: 1px solid rgba(112, 198, 255, 0.08); }
.dsh-nt-media-meta-key { color: rgba(160, 205, 240, 0.6); }
.dsh-nt-media-meta-value { color: #e6f6ff; text-align: right; word-break: break-word; }
.dsh-nt-media-viewer-path { font-size: 9px; color: rgba(166, 210, 245, 0.55); word-break: break-all; max-height: 38px; overflow: auto; }
.dsh-nt-media-viewer-actions { display: flex; gap: 6px; flex-wrap: wrap; }
.dsh-nt-media-viewer-actions a { text-decoration: none; }
.dsh-nt-node-media { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 5px; }
.dsh-nt-node-media-tile {
  padding: 0; aspect-ratio: 1;
  overflow: hidden;
  border-radius: 7px;
  border: 1px solid rgba(112, 198, 255, 0.14);
  background: #02060c;
  cursor: pointer;
}
.dsh-nt-node-media-tile:hover { border-color: rgba(111, 233, 255, 0.5); }
.dsh-nt-node-media-image { width: 100%; height: 100%; object-fit: cover; display: block; }

.dsh-nt-side {
  position: relative;
  display: flex; flex-direction: column; min-height: 0; min-width: 0;
  --nt-side-width: 440px;
  border-left: 1px solid rgba(112, 198, 255, 0.12);
  background: linear-gradient(180deg, rgba(8, 15, 27, 0.88), rgba(4, 8, 15, 0.94));
}
.dsh-nt-resizer {
  position: absolute;
  left: -4px; top: 0; bottom: 0;
  width: 8px;
  z-index: 8;
  cursor: col-resize;
  touch-action: none;
  outline: none;
}
.dsh-nt-resizer::after {
  content: "";
  position: absolute;
  left: 3px; top: 0; bottom: 0;
  width: 1px;
  background: rgba(112, 198, 255, 0.14);
  transition: background 120ms ease, box-shadow 120ms ease;
}
.dsh-nt-resizer:hover::after, .dsh-nt-resizer:focus-visible::after, .dsh-nt-resizer.is-dragging::after {
  background: rgba(111, 233, 255, 0.8);
  box-shadow: 0 0 10px rgba(80, 210, 255, 0.5);
}
.dsh-nt-tabs {
  display: flex; gap: 1px; padding: 10px 6px 0; flex: 0 0 auto;
  overflow-x: auto; scrollbar-width: none;
}
.dsh-nt-tabs::-webkit-scrollbar { display: none; }
.dsh-nt-tab {
  flex: 0 0 auto;
  height: 30px;
  padding: 0 6px;
  gap: 4px;
  font-size: 10px;
  border: 1px solid transparent;
  border-bottom: 0;
  border-radius: 9px 9px 0 0;
  background: transparent;
  color: rgba(165, 205, 240, 0.56);
  font: inherit; letter-spacing: 0.03em;
  cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
}
.dsh-nt-tab:hover { color: #e8f7ff; background: rgba(24, 44, 72, 0.34); }
.dsh-nt-tab.is-active {
  color: #eafcff;
  background: rgba(20, 42, 70, 0.55);
  border-color: rgba(112, 198, 255, 0.2);
  box-shadow: 0 -1px 0 rgba(110, 220, 255, 0.5) inset;
}
.dsh-nt-tab:disabled { opacity: 0.4; cursor: default; }
.dsh-nt-tab-count {
  min-width: 14px; height: 14px; padding: 0 3px;
  border-radius: 999px;
  background: rgba(110, 220, 255, 0.16);
  color: #a9e9ff; font-size: 8.5px; line-height: 14px; text-align: center;
}
.dsh-nt-side-body {
  flex: 1 1 auto; min-height: 0; overflow: auto; overscroll-behavior: contain;
  padding: 10px 12px 14px;
  border-top: 1px solid rgba(112, 198, 255, 0.16);
  scrollbar-width: thin;
}
.dsh-nt-side-body::-webkit-scrollbar { width: 8px; }
.dsh-nt-side-body::-webkit-scrollbar-thumb { background: rgba(110, 180, 240, 0.2); border-radius: 8px; }
.dsh-nt-side-body::-webkit-scrollbar-track { background: transparent; }
.dsh-nt-side-foot {
  flex: 0 0 auto;
  display: flex; align-items: center; gap: 6px;
  padding: 8px 13px;
  border-top: 1px solid rgba(112, 198, 255, 0.12);
  font-size: 9.5px; color: rgba(173, 210, 242, 0.62);
  letter-spacing: 0.02em;
  white-space: nowrap; overflow: hidden;
}
.dsh-nt-foot-sep { opacity: 0.5; }

.dsh-nt-flow-wrap { display: flex; flex-direction: column; gap: 9px; }
.dsh-nt-anchors { display: flex; flex-wrap: wrap; gap: 5px; }
.dsh-nt-anchor {
  display: inline-flex; align-items: center; gap: 5px;
  height: 23px; padding: 0 8px;
  border-radius: 999px;
  border: 1px solid rgba(112, 190, 245, 0.16);
  background: rgba(8, 16, 28, 0.6);
  color: rgba(205, 232, 252, 0.78);
  font: inherit; font-size: 9.5px; letter-spacing: 0.02em;
  cursor: pointer;
  transition: border-color 140ms ease, color 140ms ease, background 140ms ease;
}
.dsh-nt-anchor:hover { color: #f4fdff; border-color: var(--nt-color); background: rgba(14, 28, 48, 0.8); }
.dsh-nt-anchor-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--nt-color); box-shadow: 0 0 7px var(--nt-color); }
.dsh-nt-anchor-count {
  min-width: 14px; height: 14px; padding: 0 3px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--nt-color) 18%, transparent);
  color: var(--nt-color);
  font-size: 8.5px; line-height: 14px; text-align: center;
  font-variant-numeric: tabular-nums;
}
.dsh-nt-flow { display: flex; flex-direction: column; gap: 7px; }
.dsh-nt-flow-item {
  display: flex; gap: 9px;
  width: 100%;
  padding: 8px 9px;
  border-radius: 10px;
  border: 1px solid rgba(112, 190, 245, 0.1);
  background: rgba(9, 17, 30, 0.58);
  color: #eaf4ff;
  text-align: left;
  font: inherit;
  cursor: pointer;
  transition: border-color 140ms ease, background 140ms ease;
}
.dsh-nt-flow-item:hover { border-color: color-mix(in srgb, var(--nt-color) 48%, transparent); background: rgba(14, 27, 46, 0.8); }
.dsh-nt-flow-item.is-live { border-color: color-mix(in srgb, var(--nt-color) 54%, transparent); background: rgba(14, 30, 52, 0.78); }
.dsh-nt-flow-rail {
  flex: 0 0 auto; width: 2px; border-radius: 2px; background: var(--nt-color);
  box-shadow: 0 0 8px var(--nt-color);
  opacity: 0.65;
}
.dsh-nt-flow-main { display: flex; flex-direction: column; gap: 4px; min-width: 0; flex: 1 1 auto; }
.dsh-nt-flow-meta { display: flex; align-items: baseline; gap: 6px; font-size: 9.5px; letter-spacing: 0.05em; text-transform: uppercase; color: rgba(168, 210, 246, 0.64); }
.dsh-nt-flow-kind { color: var(--nt-color); }
.dsh-nt-flow-sector { opacity: 0.7; }
.dsh-nt-flow-time { margin-left: auto; opacity: 0.74; text-transform: none; letter-spacing: 0.02em; }
.dsh-nt-flow-text {
  font-size: 12px;
  line-height: 1.55;
  color: rgba(226, 241, 255, 0.88);
  white-space: pre-wrap;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 6;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.dsh-nt-flow-item.is-live .dsh-nt-flow-text { -webkit-line-clamp: 12; }
.dsh-nt-caret { display: inline-block; width: 6px; height: 11px; margin-left: 2px; background: #8ff0ff; vertical-align: -1px; animation: nt-blink 1s steps(2, start) infinite; }
.dsh-nt-flow-live {
  display: flex; align-items: center; gap: 7px;
  padding: 5px 4px 0;
  font-size: 10px; color: rgba(150, 240, 205, 0.82);
}
.dsh-nt-more {
  align-self: center;
  margin: 2px 0 0;
  padding: 5px 11px;
  border-radius: 999px;
  border: 1px solid rgba(112, 198, 255, 0.22);
  background: rgba(10, 22, 38, 0.62);
  color: rgba(198, 232, 255, 0.78);
  font: inherit; font-size: 10px; letter-spacing: 0.02em;
  cursor: pointer;
}
.dsh-nt-more:hover { border-color: rgba(120, 222, 255, 0.55); color: #f0fbff; }

.dsh-nt-signals { display: flex; flex-direction: column; gap: 5px; }
.dsh-nt-signal {
  display: flex; align-items: center; gap: 9px;
  width: 100%;
  padding: 7px 8px;
  border-radius: 9px;
  border: 1px solid rgba(112, 190, 245, 0.1);
  background: rgba(9, 17, 30, 0.54);
  color: #eaf4ff;
  font: inherit;
  text-align: left;
  cursor: pointer;
  transition: border-color 140ms ease, background 140ms ease;
}
.dsh-nt-signal:hover { border-color: color-mix(in srgb, var(--nt-color) 44%, transparent); background: rgba(14, 27, 46, 0.8); }
.dsh-nt-signal-icon {
  flex: 0 0 auto;
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px;
  border-radius: 7px;
  border: 1px solid color-mix(in srgb, var(--nt-color) 34%, transparent);
  color: var(--nt-color);
  background: color-mix(in srgb, var(--nt-color) 12%, transparent);
}
.dsh-nt-signal.is-error .dsh-nt-signal-icon { color: #ff9a90; border-color: rgba(255, 120, 110, 0.4); background: rgba(120, 30, 30, 0.24); }
.dsh-nt-signal-main { display: flex; flex-direction: column; gap: 1px; min-width: 0; flex: 1 1 auto; }
.dsh-nt-signal-name { font-size: 11.5px; font-weight: 560; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dsh-nt-signal-summary {
  font-size: 10px; color: rgba(186, 219, 246, 0.7);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  max-width: 100%;
}
.dsh-nt-signal-side { display: flex; flex-direction: column; align-items: flex-end; gap: 2px; flex: 0 0 auto; }
.dsh-nt-signal-sector {
  font-size: 8.5px; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--nt-color);
  opacity: 0.85;
}
.dsh-nt-signal-time { font-size: 9px; color: rgba(168, 208, 242, 0.58); }
.dsh-nt-signal.is-running .dsh-nt-signal-icon { animation: nt-glow 1.8s ease-in-out infinite; }
.dsh-nt-signal.is-running .dsh-nt-signal-name { color: #bfffe3; }

.dsh-nt-calls { display: flex; flex-direction: column; gap: 11px; }
.dsh-nt-calls-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
.dsh-nt-calls-heading { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.dsh-nt-calls-heading strong { font-size: 12px; font-weight: 600; color: #eefaff; letter-spacing: 0.01em; }
.dsh-nt-calls-heading span { font-size: 10px; line-height: 1.4; color: rgba(176, 213, 244, 0.62); }
.dsh-nt-calls-total {
  flex: 0 0 auto; padding: 2px 6px; border-radius: 999px;
  border: 1px solid rgba(112, 198, 255, 0.16);
  color: rgba(169, 230, 255, 0.82); font-size: 8.5px; white-space: nowrap;
}
.dsh-nt-call-chips { display: flex; flex-wrap: wrap; gap: 5px; }
.dsh-nt-call-chip {
  display: inline-flex; align-items: center; gap: 4px;
  height: 24px; padding: 0 7px;
  border: 1px solid rgba(112, 198, 255, 0.14);
  border-radius: 999px;
  background: rgba(9, 17, 30, 0.52);
  color: rgba(185, 219, 247, 0.68);
  font: inherit; font-size: 9.5px; letter-spacing: 0.02em;
  cursor: pointer;
}
.dsh-nt-call-chip:hover { border-color: color-mix(in srgb, var(--nt-color) 42%, transparent); color: #effbff; }
.dsh-nt-call-chip.is-active {
  border-color: color-mix(in srgb, var(--nt-color) 55%, transparent);
  background: color-mix(in srgb, var(--nt-color) 12%, rgba(9, 17, 30, 0.7));
  color: var(--nt-color);
}
.dsh-nt-call-chip svg { width: 11px; height: 11px; }
.dsh-nt-call-chip-count { color: rgba(231, 249, 255, 0.76); font-size: 8.5px; }
.dsh-nt-call-group { display: flex; flex-direction: column; gap: 5px; }
.dsh-nt-call-group-head {
  display: flex; align-items: center; gap: 6px;
  color: var(--nt-color); font-size: 8.5px; letter-spacing: 0.1em; text-transform: uppercase;
}
.dsh-nt-call-group-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; box-shadow: 0 0 10px currentColor; }
.dsh-nt-call-group-count { margin-left: auto; color: rgba(177, 218, 247, 0.6); letter-spacing: 0.02em; }
.dsh-nt-call {
  display: flex; align-items: flex-start; gap: 8px;
  width: 100%; padding: 7px 8px;
  border-radius: 9px;
  border: 1px solid color-mix(in srgb, var(--nt-color) 16%, transparent);
  background: rgba(9, 17, 30, 0.56);
  color: #eaf4ff; font: inherit; text-align: left; cursor: pointer;
  transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
}
.dsh-nt-call:hover { border-color: color-mix(in srgb, var(--nt-color) 48%, transparent); background: color-mix(in srgb, var(--nt-color) 7%, rgba(14, 27, 46, 0.86)); transform: translateX(1px); }
.dsh-nt-call.is-selected { border-color: color-mix(in srgb, var(--nt-color) 72%, transparent); background: color-mix(in srgb, var(--nt-color) 11%, rgba(14, 27, 46, 0.9)); }
.dsh-nt-call.is-error { border-color: rgba(255, 120, 110, 0.34); }
.dsh-nt-call-icon {
  flex: 0 0 auto; display: inline-flex; align-items: center; justify-content: center;
  width: 23px; height: 23px; margin-top: 1px;
  border-radius: 7px;
  border: 1px solid color-mix(in srgb, var(--nt-color) 36%, transparent);
  color: var(--nt-color);
  background: color-mix(in srgb, var(--nt-color) 11%, transparent);
}
.dsh-nt-call.is-running .dsh-nt-call-icon { animation: nt-glow 1.8s ease-in-out infinite; }
.dsh-nt-call-main { display: flex; flex-direction: column; gap: 2px; min-width: 0; flex: 1 1 auto; }
.dsh-nt-call-top { display: flex; align-items: center; gap: 5px; min-width: 0; }
.dsh-nt-call-badge {
  flex: 0 0 auto; padding: 1px 4px; border-radius: 4px;
  border: 1px solid color-mix(in srgb, var(--nt-color) 36%, transparent);
  color: var(--nt-color); font-size: 7px; line-height: 1.25; letter-spacing: 0.1em; text-transform: uppercase;
}
.dsh-nt-call-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; font-weight: 550; color: #f0fbff; }
.dsh-nt-call-summary { font-size: 9.5px; line-height: 1.35; color: rgba(176, 213, 244, 0.66); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dsh-nt-call-side { flex: 0 0 auto; display: flex; flex-direction: column; align-items: flex-end; gap: 2px; }
.dsh-nt-call-time { color: rgba(168, 208, 242, 0.52); font-size: 8.5px; }
.dsh-nt-call-state { color: #ff9a90; font-size: 8px; letter-spacing: 0.04em; }
.dsh-nt-call-clear {
  margin-top: 4px; padding: 3px 8px;
  border: 1px solid rgba(112, 198, 255, 0.22);
  border-radius: 999px;
  background: rgba(9, 17, 30, 0.55);
  color: rgba(196, 232, 255, 0.82);
  font: inherit; font-size: 9px; cursor: pointer;
}
.dsh-nt-call-clear:hover { border-color: rgba(112, 198, 255, 0.5); color: #f0fbff; }

.dsh-nt-empty-panel {
  height: 100%;
  min-height: 220px;
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;
  text-align: center;
  color: rgba(180, 215, 245, 0.58);
  font-size: 11.5px;
  line-height: 1.55;
  padding: 20px;
}
.dsh-nt-empty-panel strong { color: #e9f6ff; font-size: 13px; font-weight: 560; }
.dsh-nt-empty-panel.is-compact { min-height: 128px; padding: 14px 8px; }
.dsh-nt-empty-orb {
  display: inline-flex; align-items: center; justify-content: center;
  width: 56px; height: 56px;
  border-radius: 50%;
  border: 1px solid rgba(112, 198, 255, 0.18);
  background: radial-gradient(circle at 50% 42%, rgba(60, 160, 255, 0.18), rgba(6, 12, 22, 0.2) 70%);
  color: #7fe4ff;
  box-shadow: 0 0 34px rgba(40, 140, 255, 0.16) inset;
  margin-bottom: 4px;
}

.dsh-nt-node { display: flex; flex-direction: column; gap: 11px; }
.dsh-nt-node-hero { display: flex; align-items: flex-start; gap: 10px; }
.dsh-nt-node-orb {
  width: 14px; height: 14px; margin-top: 4px; border-radius: 50%;
  background: var(--nt-color);
  box-shadow: 0 0 16px var(--nt-color);
  flex: 0 0 auto;
}
.dsh-nt-node-hero-text { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.dsh-nt-node-kicker { font-size: 9px; letter-spacing: 0.13em; text-transform: uppercase; color: var(--nt-color); }
.dsh-nt-node-family {
  display: inline-flex; align-items: center; gap: 4px;
  color: var(--nt-color);
}
.dsh-nt-node-family svg { width: 11px; height: 11px; }
.dsh-nt-node-hero h3 { margin: 0; font-size: 14px; font-weight: 600; line-height: 1.28; color: #f0fbff; word-break: break-word; }
.dsh-nt-node-sub { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; font-size: 10px; color: rgba(183, 218, 246, 0.7); }
.dsh-nt-node-dot { opacity: 0.5; }
.dsh-nt-node-status { color: rgba(150, 240, 205, 0.8); }
.dsh-nt-node-status.error { color: #ff9a90; }
.dsh-nt-node-status.running { color: #ffd479; }
.dsh-nt-node-note {
  padding: 7px 9px;
  border-radius: 9px;
  border: 1px solid rgba(255, 207, 107, 0.26);
  background: rgba(92, 70, 20, 0.22);
  color: rgba(255, 226, 168, 0.82);
  font-size: 10.5px;
  line-height: 1.5;
}
.dsh-nt-node-error {
  display: flex; align-items: center; gap: 7px;
  padding: 7px 9px;
  border-radius: 9px;
  border: 1px solid rgba(255, 120, 110, 0.3);
  background: rgba(90, 22, 22, 0.28);
  color: #ffb6ad;
  font-size: 11px;
}
.dsh-nt-node-meta {
  display: flex; flex-direction: column; gap: 1px;
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid rgba(112, 198, 255, 0.12);
  background: rgba(8, 16, 28, 0.55);
}
.dsh-nt-node-meta-row {
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  padding: 4px 0;
  border-bottom: 1px solid rgba(112, 198, 255, 0.07);
  font-size: 10.5px;
  color: rgba(183, 218, 246, 0.68);
}
.dsh-nt-node-meta-row:last-child { border-bottom: 0; }
.dsh-nt-node-meta-value { color: #e6f6ff; text-align: right; word-break: break-word; font-variant-numeric: tabular-nums; }
.dsh-nt-node-section { display: flex; flex-direction: column; gap: 5px; }
.dsh-nt-node-section h4 {
  margin: 0; font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase;
  color: rgba(155, 205, 245, 0.5);
}
.dsh-nt-pre {
  margin: 0;
  max-height: 236px;
  overflow: auto;
  overscroll-behavior: contain;
  padding: 9px 10px;
  border-radius: 9px;
  border: 1px solid rgba(112, 198, 255, 0.12);
  background: rgba(4, 9, 16, 0.72);
  color: rgba(216, 238, 255, 0.86);
  font: 10.5px/1.55 ui-monospace, "SF Mono", SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap;
  word-break: break-word;
}
.dsh-nt-node-text {
  max-height: 330px;
  overflow: auto;
  overscroll-behavior: contain;
  padding: 9px 10px;
  border-radius: 9px;
  border: 1px solid rgba(112, 198, 255, 0.12);
  background: rgba(6, 13, 23, 0.6);
  color: rgba(228, 242, 255, 0.88);
  font-size: 11.5px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}
.dsh-nt-links { display: flex; flex-wrap: wrap; gap: 5px; }
.dsh-nt-link-chip {
  display: inline-flex; align-items: center; gap: 6px;
  max-width: 100%;
  padding: 5px 8px;
  border-radius: 999px;
  border: 1px solid rgba(112, 198, 255, 0.16);
  background: rgba(10, 20, 34, 0.6);
  color: rgba(212, 238, 255, 0.82);
  font: inherit; font-size: 10px;
  cursor: pointer;
}
.dsh-nt-link-chip:hover { border-color: var(--nt-color); color: #f2fcff; }
.dsh-nt-link-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--nt-color); box-shadow: 0 0 7px var(--nt-color); }
.dsh-nt-node-actions { display: flex; gap: 6px; padding-top: 2px; }

.dsh-nt-spinner { animation: nt-spin 0.9s linear infinite; }
.dsh-nt-pulse { animation: nt-pulse 1.7s ease-in-out infinite; }


/* ---- Mode, language, fullscreen ---- */
.dsh-nt-mode {
  display: inline-flex;
  flex: 0 0 auto;
  gap: 2px;
  padding: 2px;
  border-radius: 9px;
  border: 1px solid rgba(120, 200, 255, 0.16);
  background: rgba(8, 17, 30, 0.7);
}
.dsh-nt-mode-button {
  height: 24px;
  padding: 0 9px;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: rgba(178, 214, 245, 0.66);
  font: inherit;
  font-size: 10.5px;
  letter-spacing: 0.02em;
  cursor: pointer;
  transition: color 140ms ease, background 140ms ease, box-shadow 140ms ease;
}
.dsh-nt-mode-button:hover { color: #eafcff; background: rgba(26, 56, 88, 0.5); }
.dsh-nt-mode-button.is-active {
  color: #f0fcff;
  background: linear-gradient(180deg, rgba(28, 82, 124, 0.8), rgba(16, 50, 80, 0.85));
  box-shadow: 0 0 0 1px rgba(110, 220, 255, 0.28) inset, 0 0 14px rgba(60, 180, 255, 0.14);
}
.dsh-nt-mode-button:disabled { opacity: 0.38; cursor: default; }

.dsh-nt-lang { position: relative; z-index: 6; }
.dsh-nt-lang-button {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 28px;
  padding: 0 8px;
  border-radius: 8px;
  border: 1px solid rgba(120, 200, 255, 0.16);
  background: rgba(10, 20, 36, 0.66);
  color: rgba(200, 232, 255, 0.78);
  font: inherit;
  cursor: pointer;
}
.dsh-nt-lang-button:hover { color: #ecfaff; border-color: rgba(110, 220, 255, 0.5); }
.dsh-nt-lang-code { font-size: 9px; letter-spacing: 0.08em; }
.dsh-nt-lang-pop {
  position: absolute;
  right: 0;
  top: calc(100% + 8px);
  width: 224px;
  padding: 6px;
  border-radius: 12px;
  border: 1px solid rgba(110, 205, 255, 0.26);
  background: linear-gradient(180deg, rgba(10, 20, 36, 0.99), rgba(6, 11, 20, 0.99));
  box-shadow: 0 26px 70px rgba(0, 0, 0, 0.7), 0 0 40px rgba(40, 140, 255, 0.1);
  z-index: 40;
  animation: nt-pop 140ms ease-out both;
}
.dsh-nt-lang-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 7px 8px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: transparent;
  color: #eaf4ff;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.dsh-nt-lang-item:hover { background: rgba(26, 52, 84, 0.55); }
.dsh-nt-lang-item.is-active { border-color: rgba(110, 220, 255, 0.3); background: rgba(20, 44, 74, 0.6); }
.dsh-nt-lang-native { flex: 1 1 auto; font-size: 12px; }
.dsh-nt-lang-id { font-size: 8.5px; letter-spacing: 0.06em; color: rgba(165, 205, 240, 0.45); }
.dsh-nt-lang-check { display: inline-flex; color: #6fe9ff; }

.dsh-nt-overlay.is-full { padding: 0; }
.dsh-nt-console.is-full { width: 100vw; height: 100vh; max-width: none; border-radius: 0; border: 0; }
.dsh-nt-overlay.is-mac.is-full .dsh-nt-head { padding-top: 38px; }
.dsh-nt-icon-button.is-on {
  color: #a9ecff;
  border-color: rgba(110, 220, 255, 0.48);
  background: rgba(18, 48, 76, 0.82);
  box-shadow: 0 0 14px rgba(64, 180, 255, 0.2);
}

/* ---- Classes ---- */
.dsh-nt-class-wrap { position: relative; }
.dsh-nt-class-button {
  width: auto;
  min-width: 28px;
  gap: 5px;
  padding: 0 8px;
  font-size: 10px;
  letter-spacing: 0.02em;
}
.dsh-nt-class-button-label { white-space: nowrap; }
.dsh-nt-class-dot {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  box-shadow: 0 0 7px currentColor;
  flex: 0 0 auto;
}
.dsh-nt-class-active {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--nt-color);
}
.dsh-nt-class-menu {
  position: absolute;
  top: 36px;
  right: 0;
  width: min(292px, calc(100vw - 40px));
  max-height: min(430px, 62vh);
  overflow: auto;
  overscroll-behavior: contain;
  padding: 6px;
  border-radius: 12px;
  border: 1px solid rgba(110, 205, 255, 0.26);
  background: linear-gradient(180deg, rgba(10, 20, 36, 0.99), rgba(6, 11, 20, 0.99));
  box-shadow: 0 26px 70px rgba(0, 0, 0, 0.7), 0 0 40px rgba(40, 140, 255, 0.1);
  z-index: 40;
  animation: nt-pop 140ms ease-out both;
  scrollbar-width: thin;
}
.dsh-nt-class-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  width: 100%;
  padding: 7px 8px;
  border: 1px solid transparent;
  border-radius: 9px;
  background: transparent;
  color: #eaf4ff;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.dsh-nt-class-row:hover { background: rgba(26, 52, 84, 0.55); }
.dsh-nt-class-row.is-active {
  border-color: color-mix(in srgb, var(--nt-color) 42%, transparent);
  background: color-mix(in srgb, var(--nt-color) 10%, rgba(10, 22, 38, 0.72));
}
.dsh-nt-class-row-dot {
  width: 7px; height: 7px;
  margin-top: 4px;
  border-radius: 50%;
  box-shadow: 0 0 8px currentColor;
  flex: 0 0 auto;
}
.dsh-nt-class-row-dot.is-all { background: linear-gradient(135deg, #5fe3ff, #ffb86b); }
.dsh-nt-class-row-main { display: flex; flex-direction: column; gap: 2px; flex: 1 1 auto; min-width: 0; }
.dsh-nt-class-row-name { font-size: 11.5px; color: #eefaff; }
.dsh-nt-class-row-hint { font-size: 9.5px; line-height: 1.42; color: rgba(188, 220, 246, 0.64); }
.dsh-nt-class-count {
  margin-top: 2px;
  font-size: 9px;
  color: rgba(176, 214, 245, 0.66);
  font-variant-numeric: tabular-nums;
}
.dsh-nt-class-check { display: inline-flex; margin-top: 2px; color: #6fe9ff; }

/* ---- Sector legend, hover note, guide ---- */
.dsh-nt-project-legend { pointer-events: none; }
.dsh-nt-legend-chip.is-static { pointer-events: none; cursor: default; }
.dsh-nt-legend-info { padding: 0 7px; }
.dsh-nt-sector-note {
  position: absolute;
  left: 14px;
  bottom: 44px;
  width: min(366px, calc(100% - 28px));
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid color-mix(in srgb, var(--nt-color) 42%, transparent);
  background: linear-gradient(160deg, rgba(8, 18, 32, 0.96), rgba(5, 11, 20, 0.97));
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.55), 0 0 26px color-mix(in srgb, var(--nt-color) 16%, transparent);
  z-index: 7;
  pointer-events: none;
  animation: nt-pop 130ms ease-out both;
}
.dsh-nt-sector-note-head { display: flex; align-items: center; gap: 7px; }
.dsh-nt-sector-note-head strong { font-size: 11.5px; color: #f0fbff; font-weight: 600; }
.dsh-nt-sector-note-count {
  margin-left: auto;
  min-width: 18px; height: 16px; padding: 0 5px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--nt-color) 18%, transparent);
  color: var(--nt-color);
  font-size: 9px; line-height: 16px; text-align: center;
  font-variant-numeric: tabular-nums;
}
.dsh-nt-sector-note p {
  margin: 6px 0 0;
  font-size: 11px;
  line-height: 1.5;
  color: rgba(208, 232, 250, 0.8);
}
.dsh-nt-sector-guide {
  position: absolute;
  left: 14px;
  bottom: 44px;
  width: min(392px, calc(100% - 28px));
  max-height: min(340px, 56vh);
  overflow: auto;
  overscroll-behavior: contain;
  padding: 8px;
  border-radius: 13px;
  border: 1px solid rgba(110, 205, 255, 0.24);
  background: linear-gradient(170deg, rgba(8, 18, 32, 0.98), rgba(5, 10, 18, 0.99));
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.65), 0 0 38px rgba(40, 140, 255, 0.1);
  z-index: 8;
  animation: nt-pop 160ms ease-out both;
  scrollbar-width: thin;
}
.dsh-nt-sector-guide-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 2px 4px 7px;
}
.dsh-nt-sector-guide-head strong { font-size: 11.5px; color: #eafcff; font-weight: 600; }
.dsh-nt-sector-guide-row {
  display: flex; align-items: flex-start; gap: 8px;
  width: 100%;
  padding: 7px;
  border: 1px solid transparent;
  border-radius: 9px;
  background: transparent;
  color: #eaf4ff;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.dsh-nt-sector-guide-row:hover { background: rgba(24, 48, 78, 0.5); }
.dsh-nt-sector-guide-row.is-active { border-color: color-mix(in srgb, var(--nt-color) 40%, transparent); background: color-mix(in srgb, var(--nt-color) 10%, rgba(10, 22, 38, 0.7)); }
.dsh-nt-sector-guide-row .dsh-nt-legend-dot { margin-top: 4px; flex: 0 0 auto; }
.dsh-nt-sector-guide-text { flex: 1 1 auto; display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.dsh-nt-sector-guide-name { font-size: 11.5px; color: #eefaff; }
.dsh-nt-sector-guide-hint { font-size: 10px; line-height: 1.45; color: rgba(190, 222, 246, 0.68); }
.dsh-nt-sector-guide-row .dsh-nt-legend-count { margin-top: 3px; flex: 0 0 auto; }

/* ---- Hover tooltip ---- */
.dsh-nt-tip {
  position: absolute;
  width: 296px;
  padding: 9px 11px;
  border-radius: 11px;
  border: 1px solid rgba(110, 205, 255, 0.28);
  background: linear-gradient(160deg, rgba(7, 16, 28, 0.97), rgba(4, 9, 17, 0.98));
  box-shadow: 0 20px 48px rgba(0, 0, 0, 0.62), 0 0 26px rgba(40, 150, 255, 0.12);
  z-index: 9;
  pointer-events: none;
  animation: nt-pop 120ms ease-out both;
}
.dsh-nt-tip-head { display: flex; align-items: center; gap: 6px; }
.dsh-nt-tip-dot { width: 7px; height: 7px; border-radius: 50%; box-shadow: 0 0 8px currentColor; flex: 0 0 auto; }
.dsh-nt-tip-kind { font-size: 8.5px; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(170, 212, 245, 0.66); }
.dsh-nt-tip-sector { font-size: 8.5px; letter-spacing: 0.04em; color: rgba(140, 200, 240, 0.52); }
.dsh-nt-tip-family { font-size: 8.5px; letter-spacing: 0.04em; color: rgba(111, 240, 255, 0.72); }
.dsh-nt-tip.is-call { border-color: rgba(111, 240, 255, 0.26); }
.dsh-nt-tip.is-skill { border-color: rgba(214, 180, 255, 0.3); }
.dsh-nt-tip.is-mcp { border-color: rgba(111, 240, 255, 0.3); }
.dsh-nt-tip-status { margin-left: auto; font-size: 8.5px; color: rgba(150, 240, 205, 0.78); }
.dsh-nt-tip-status.error { color: #ff9a90; }
.dsh-nt-tip-status.running { color: #ffd479; }
.dsh-nt-tip-title {
  margin-top: 4px;
  font-size: 12.5px;
  font-weight: 600;
  line-height: 1.35;
  color: #f0fbff;
  word-break: break-word;
}
.dsh-nt-tip-text {
  margin-top: 4px;
  font-size: 10.5px;
  line-height: 1.5;
  color: rgba(203, 228, 249, 0.76);
  white-space: pre-wrap;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* ---- Filter ---- */
.dsh-nt-filter {
  display: flex;
  align-items: center;
  gap: 7px;
  flex: 0 0 auto;
  margin: 8px 12px 0;
  padding: 5px 8px;
  border-radius: 9px;
  border: 1px solid rgba(120, 200, 255, 0.14);
  background: rgba(6, 13, 24, 0.72);
  color: rgba(165, 210, 245, 0.58);
}
.dsh-nt-filter input {
  flex: 1 1 auto;
  min-width: 0;
  border: 0;
  background: transparent;
  color: #eaf4ff;
  font: inherit;
  font-size: 11px;
  outline: none;
}
.dsh-nt-filter input::placeholder { color: rgba(150, 195, 235, 0.42); }
.dsh-nt-filter-clear {
  display: inline-flex; align-items: center; justify-content: center;
  width: 20px; height: 20px; padding: 0;
  border: 0; border-radius: 6px;
  background: rgba(110, 200, 255, 0.1);
  color: rgba(190, 228, 255, 0.7);
  cursor: pointer;
}
.dsh-nt-filter-clear:hover { color: #f0fbff; background: rgba(110, 200, 255, 0.2); }

/* ---- Project side panel ---- */
.dsh-nt-project { display: flex; flex-direction: column; gap: 9px; }
.dsh-nt-project-summary {
  padding: 10px 11px;
  border-radius: 11px;
  border: 1px solid rgba(112, 198, 255, 0.14);
  background: linear-gradient(160deg, rgba(12, 26, 44, 0.7), rgba(6, 13, 24, 0.6));
}
.dsh-nt-project-title { font-size: 13px; font-weight: 600; color: #f0fbff; word-break: break-word; }
.dsh-nt-project-path {
  margin-top: 2px;
  font-size: 10px;
  color: rgba(172, 212, 246, 0.6);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.dsh-nt-project-stats { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.dsh-nt-project-stat {
  padding: 3px 7px;
  border-radius: 999px;
  border: 1px solid rgba(120, 200, 255, 0.18);
  background: rgba(8, 18, 32, 0.6);
  color: rgba(190, 224, 250, 0.78);
  font-size: 9px;
  letter-spacing: 0.02em;
}
.dsh-nt-project-stat.is-links { border-color: rgba(183, 139, 255, 0.28); color: #ceb9ff; }
.dsh-nt-project-stat.is-running { display: inline-flex; align-items: center; gap: 5px; border-color: rgba(71, 240, 168, 0.3); color: #8df5c4; }
.dsh-nt-project-search { margin: 0; }
.dsh-nt-project-list { display: flex; flex-direction: column; gap: 4px; }
.dsh-nt-project-row {
  display: flex;
  align-items: center;
  gap: 9px;
  width: 100%;
  padding: 8px;
  border-radius: 9px;
  border: 1px solid rgba(112, 190, 245, 0.1);
  background: rgba(9, 17, 30, 0.54);
  color: #eaf4ff;
  font: inherit;
  text-align: left;
  cursor: pointer;
  transition: border-color 140ms ease, background 140ms ease;
}
.dsh-nt-project-row:hover { border-color: rgba(110, 220, 255, 0.4); background: rgba(14, 28, 48, 0.82); }
.dsh-nt-project-row.is-linked { border-color: rgba(183, 139, 255, 0.24); }
.dsh-nt-project-row.is-muted { opacity: 0.55; }
.dsh-nt-project-row-main { flex: 1 1 auto; display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.dsh-nt-project-row-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11.5px; }
.dsh-nt-project-row-meta {
  display: flex; align-items: center; gap: 6px;
  font-size: 9.5px; color: rgba(172, 210, 244, 0.58);
  overflow: hidden; white-space: nowrap;
}
.dsh-nt-project-badge {
  padding: 1px 5px;
  border-radius: 999px;
  border: 1px solid rgba(183, 139, 255, 0.34);
  color: #cdb8ff;
  font-size: 8px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.dsh-nt-project-parent { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dsh-nt-project-open { color: rgba(160, 215, 250, 0.5); flex: 0 0 auto; }

/* ---- Node links direction ---- */
.dsh-nt-link-arrow { color: var(--nt-color); font-size: 9px; line-height: 1; }
.dsh-nt-links-empty { font-size: 10px; color: rgba(170, 210, 245, 0.45); padding: 2px 0; }
.dsh-nt-node-actions { flex-wrap: wrap; }

@keyframes nt-spin { to { transform: rotate(360deg); } }
@keyframes nt-pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.45; transform: scale(0.82); } }
@keyframes nt-glow { 0%, 100% { box-shadow: 0 0 0 rgba(80, 220, 255, 0); } 50% { box-shadow: 0 0 14px color-mix(in srgb, var(--nt-color) 45%, transparent); } }
@keyframes nt-blink { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0; } }
@keyframes nt-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes nt-rise { from { opacity: 0; transform: translateY(14px) scale(0.985); } to { opacity: 1; transform: translateY(0) scale(1); } }
@keyframes nt-pop { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }
@keyframes nt-toast-in { from { opacity: 0; transform: translateY(-7px) scale(0.98); } to { opacity: 1; transform: translateY(0) scale(1); } }

@media (max-width: 1180px) {
  .dsh-nt-model-chip { display: none; }
}
@media (max-width: 980px) {
  .dsh-nt-live { display: none; }
  .dsh-nt-class-button-label { display: none; }
}
@media (max-width: 1060px) {
  .dsh-nt-body { grid-template-columns: minmax(0, 1fr) min(var(--nt-side-width, 440px), 42vw); }
  .dsh-nt-resizer { display: none; }
  .dsh-nt-model-chip { max-width: 150px; }
}
@media (max-width: 760px) {
  .dsh-nt-legend { gap: 4px; }
  .dsh-nt-legend-chip { padding: 0 6px; gap: 4px; font-size: 9.5px; }
  .dsh-nt-legend-info { padding: 0 5px; }
}
@media (max-width: 900px) {
  .dsh-nt-overlay { padding: 10px; }
  .dsh-nt-console { width: 100%; height: min(920px, 94vh); }
  .dsh-nt-body { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1.25fr) minmax(230px, 0.9fr); }
  .dsh-nt-side { border-left: 0; border-top: 1px solid rgba(112, 198, 255, 0.14); }
  .dsh-nt-brand-text span { display: none; }
  .dsh-nt-head { gap: 10px; flex-wrap: wrap; }
  .dsh-nt-picker { order: 3; width: 100%; }
  .dsh-nt-picker-button { width: 100%; max-width: none; }
  .dsh-nt-picker-pop { width: 100%; }
  .dsh-nt-session-list { max-height: 42vh; }
  .dsh-nt-head-right { order: 2; }
  .dsh-nt-stage-hint { display: none; }
}
@media (max-width: 560px) {
  .dsh-nt-overlay { padding: 0; }
  .dsh-nt-console { width: 100vw; height: 100vh; border-radius: 0; border: 0; }
  .dsh-nt-overlay.is-mac .dsh-nt-head { padding-top: 38px; }
  .dsh-nt-brand-text h2 { font-size: 14px; }
  .dsh-nt-model-chip { display: none; }
  .dsh-nt-legend-chip { padding: 0 7px; }
  .dsh-nt-legend-label { display: none; }
  .dsh-nt-toasts { top: 50px; }
}
@media (prefers-reduced-motion: reduce) {
  .dsh-nt-console, .dsh-nt-backdrop, .dsh-nt-toast, .dsh-nt-picker-pop { animation: none; }
  .dsh-nt-live-dot, .dsh-nt-pill-dot, .dsh-nt-session-state.is-running { animation: none; }
}
`

    /* ------------------------------------------------------------------ *
     *  Plugin entry
     * ------------------------------------------------------------------ */

    function apply(ctx) {
      const uiStore = createUiStore()
      const runtime = createRuntime(ctx)

      ctx.effect(() => {
        const style = document.createElement('style')
        style.setAttribute('data-plugin', '@local/neural-thoughts')
        style.setAttribute('data-plugin-css', 'neural-thoughts')
        style.textContent = CSS
        document.head.appendChild(style)
        return () => { style.remove() }
      }, 'neural-thoughts: styles')

      ctx.effect(() => () => runtime.dispose(), 'neural-thoughts: runtime')

      ctx.effect(() => ctx.slots.inject('conversation.composer.dock', () => ctx.slots.register({
        name: 'conversation.composer.dock',
        id: 'neural-thoughts',
        order: 6,
      }, function NeuralPillSlot(props) {
        return h(NeuralPill, Object.assign({}, props, { ctx, runtime, uiStore }))
      })), 'neural-thoughts: composer pill')

      ctx.effect(() => ctx.slots.inject('shell.overlay', () => ctx.slots.register({
        name: 'shell.overlay',
        id: 'neural-thoughts-panel',
        order: 40,
      }, function NeuralPanelSlot(props) {
        const base = { ctx, runtime, uiStore, useWorkspaces: props.useWorkspaces }
        return typeof props.useWorkspaces === 'function'
          ? h(NeuralPanelConnected, base)
          : h(NeuralPanel, Object.assign({}, base, { workspaceSnapshot: null }))
      })), 'neural-thoughts: overlay panel')
    }

    return {
      inject: ['slots', 'sessions', 'locale', 'uiConversation'],
      apply,
    }
  },
})
