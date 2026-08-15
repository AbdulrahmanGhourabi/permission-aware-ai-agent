import { useState } from 'react'
import { supabase } from '../supabaseClient'

const API_URL = 'http://localhost:8000'
const REQUEST_TIMEOUT_MS = 30000
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024 // 10 MB
const ALLOWED_UPLOAD_EXTENSIONS = ['.txt', '.pdf']

function Chat({ onSessionExpired }) {
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [file, setFile] = useState(null)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [uploading, setUploading] = useState(false)

  async function getAccessToken() {
    let { data, error: sessionError } = await supabase.auth.getSession()

    // If the token is missing or close to expiry, try a refresh before
    // giving up - avoids surfacing a raw 401 for a routine expired session.
    const expiresAt = data?.session?.expires_at
    const isExpiringSoon = expiresAt && expiresAt * 1000 - Date.now() < 60000

    if (sessionError || !data?.session || isExpiringSoon) {
      const refreshed = await supabase.auth.refreshSession()
      if (refreshed.error || !refreshed.data?.session) {
        throw new Error('SESSION_EXPIRED')
      }
      return refreshed.data.session.access_token
    }

    return data.session.access_token
  }

  // Forces a refresh regardless of the token's current expiry, for the
  // retry-after-401 path below - the proactive check in getAccessToken()
  // can still miss a token that was valid when the request was built but
  // expired in transit (slow network, backgrounded tab, etc), so this is
  // a separate unconditional refresh used only after the server has
  // already told us the token didn't work.
  async function forceRefreshAccessToken() {
    const refreshed = await supabase.auth.refreshSession()
    if (refreshed.error || !refreshed.data?.session) {
      throw new Error('SESSION_EXPIRED')
    }
    return refreshed.data.session.access_token
  }

  async function fetchWithTimeout(url, options) {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

    try {
      return await fetch(url, { ...options, signal: controller.signal })
    } catch (err) {
      if (err.name === 'AbortError') {
        throw new Error('Request timed out. Please try again.')
      }
      throw err
    } finally {
      clearTimeout(timeoutId)
    }
  }

  // Shared request path for both /chat and /ingest: attaches the current
  // token, and if the server comes back with 401 (token died in transit,
  // after the proactive check in getAccessToken() already passed), tries
  // exactly one forced refresh + retry before giving up. Previously each
  // caller duplicated the token-fetch + 401-branch logic independently,
  // which meant adding retry behavior would have meant patching it twice
  // and risking the two copies drifting apart later.
  //
  // buildOptions is a function so the caller can rebuild the request body
  // fresh on retry (needed for FormData, which can only be read/sent
  // once) rather than reusing a single already-sent options object.
  async function authedFetch(url, buildOptions) {
    let token = await getAccessToken()
    let res = await fetchWithTimeout(url, buildOptions(token))

    if (res.status === 401) {
      token = await forceRefreshAccessToken()
      res = await fetchWithTimeout(url, buildOptions(token))
    }

    return res
  }

  function handleAuthFailure() {
    setError('Your session has expired. Please sign in again.')
    if (onSessionExpired) onSessionExpired()
  }

  async function handleAsk(e) {
    e.preventDefault()
    if (!question.trim() || loading) return

    setLoading(true)
    setError(null)

    const userMessage = { role: 'user', content: question }
    const messageIndex = messages.length
    setMessages((prev) => [...prev, userMessage])
    setQuestion('')

    try {
      const res = await authedFetch(`${API_URL}/chat`, (token) => ({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ question: userMessage.content }),
      }))

      if (res.status === 401) {
        handleAuthFailure()
        return
      }

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.detail || `Request failed (${res.status})`)
      }

      const data = await res.json()
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.answer, chunksUsed: data.chunks_used },
      ])
    } catch (err) {
      if (err.message === 'SESSION_EXPIRED') {
        handleAuthFailure()
      } else {
        setError(err.message)
        // Mark the user's message as failed to send rather than leaving
        // it looking identical to a successfully sent one - previously
        // a failed request left the message in the list with no visual
        // difference from a message that actually reached the server.
        setMessages((prev) =>
          prev.map((m, i) => (i === messageIndex ? { ...m, failed: true } : m))
        )
      }
    } finally {
      setLoading(false)
    }
  }

  function validateFile(candidate) {
    const name = candidate.name.toLowerCase()
    const hasAllowedExtension = ALLOWED_UPLOAD_EXTENSIONS.some((ext) => name.endsWith(ext))

    if (!hasAllowedExtension) {
      return `Unsupported file type. Allowed: ${ALLOWED_UPLOAD_EXTENSIONS.join(', ')}`
    }
    if (candidate.size > MAX_UPLOAD_BYTES) {
      return `File is too large. Max size is ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB.`
    }
    return null
  }

  function handleFileChange(e) {
    const candidate = e.target.files[0]
    if (!candidate) {
      setFile(null)
      return
    }

    const validationError = validateFile(candidate)
    if (validationError) {
      setUploadStatus({ ok: false, message: validationError })
      setFile(null)
      e.target.value = ''
      return
    }

    setUploadStatus(null)
    setFile(candidate)
  }

  async function handleUpload(e) {
    e.preventDefault()
    if (uploading) return

    if (!file) {
      setUploadStatus({ ok: false, message: 'Choose a file to upload.' })
      return
    }

    setUploading(true)
    setUploadStatus({ ok: null, message: 'Uploading...' })

    try {
      // buildOptions rebuilds a fresh FormData each call - required
      // because authedFetch may call this twice (initial + retry after
      // 401) and a FormData/file body can only be sent once.
      const res = await authedFetch(`${API_URL}/ingest`, (token) => {
        const formData = new FormData()
        formData.append('file', file)
        return {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
          },
          body: formData,
        }
      })

      if (res.status === 401) {
        handleAuthFailure()
        return
      }

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.detail || `Upload failed (${res.status})`)
      }

      const data = await res.json()
      setUploadStatus({
        ok: true,
        message: `Uploaded. ${data.chunks_stored} chunks stored.`,
      })
      setFile(null)
    } catch (err) {
      if (err.message === 'SESSION_EXPIRED') {
        handleAuthFailure()
      } else {
        setUploadStatus({ ok: false, message: err.message })
      }
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="chat-layout">
      <section className="upload-panel">
        <h2>Upload a document</h2>
        <form onSubmit={handleUpload} className="upload-form">
          <input
            type="file"
            onChange={handleFileChange}
            accept={ALLOWED_UPLOAD_EXTENSIONS.join(',')}
            disabled={uploading}
          />
          <button type="submit" disabled={uploading || !file}>
            {uploading ? 'Uploading...' : 'Upload'}
          </button>
        </form>
        {uploadStatus && (
          <p className={uploadStatus.ok === false ? 'status-error' : 'status-ok'}>
            {uploadStatus.message}
          </p>
        )}
      </section>

      <section className="chat-panel">
        <div className="messages">
          {messages.length === 0 && (
            <p className="empty-state">Ask a question about a document you've uploaded.</p>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`message message-${m.role}${m.failed ? ' message-failed' : ''}`}>
              <span className="message-role">{m.role === 'user' ? 'You' : 'Assistant'}</span>
              <p>{m.content}</p>
              {m.failed && <span className="message-meta status-error">Failed to send</span>}
              {m.role === 'assistant' && (
                <span className="message-meta">{m.chunksUsed} source chunk(s) used</span>
              )}
            </div>
          ))}
          {loading && <div className="message message-assistant"><p>Thinking...</p></div>}
        </div>

        {error && <p className="status-error">{error}</p>}

        <form onSubmit={handleAsk} className="ask-form">
          <input
            type="text"
            placeholder="Ask a question..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !question.trim()}>Send</button>
        </form>
      </section>
    </div>
  )
}

export default Chat