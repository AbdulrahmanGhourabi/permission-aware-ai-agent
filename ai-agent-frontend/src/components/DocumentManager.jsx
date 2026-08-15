import { useState, useEffect, useCallback } from 'react'
import { supabase } from '../supabaseClient'

const API_URL = 'http://localhost:8000'
const REQUEST_TIMEOUT_MS = 30000

// Mirrors Chat.jsx's token/retry pattern so both components behave
// identically on session expiry and transient 401s, rather than each
// component reinventing its own auth-refresh logic.
function DocumentManager({ onSessionExpired }) {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')

  // Per-document share UI state, keyed by document id, so opening the
  // share form on one document doesn't affect another's input/state.
  const [openShareFormId, setOpenShareFormId] = useState(null)
  const [shareEmail, setShareEmail] = useState('')
  const [sharePermission, setSharePermission] = useState('viewer')
  const [shareStatus, setShareStatus] = useState(null)
  const [sharing, setSharing] = useState(false)

  // shares[documentId] = [{email, permission}, ...] - loaded once for
  // every OWNED document right after the document list loads, so the
  // "Shared with" list is always visible without needing a click, and
  // doesn't disappear/reset after a share or revoke action the way a
  // per-click-expand pattern would.
  const [shares, setShares] = useState({})

  async function getAccessToken() {
    let { data, error: sessionError } = await supabase.auth.getSession()

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

  async function loadShares(documentId) {
    try {
      const res = await authedFetch(`${API_URL}/documents/${documentId}/shares`, (token) => ({
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
      }))

      if (!res.ok) {
        setShares((prev) => ({ ...prev, [documentId]: [] }))
        return
      }

      const data = await res.json()
      setShares((prev) => ({ ...prev, [documentId]: data.shares }))
    } catch (err) {
      setShares((prev) => ({ ...prev, [documentId]: [] }))
    }
  }

  const loadDocuments = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await authedFetch(`${API_URL}/documents`, (token) => ({
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
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
      setDocuments(data)

      data.filter((doc) => doc.permission === 'owner').forEach((doc) => loadShares(doc.id))
    } catch (err) {
      if (err.message === 'SESSION_EXPIRED') {
        handleAuthFailure()
      } else {
        setError(err.message)
      }
    } finally {
      setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    loadDocuments()
  }, [loadDocuments])

  function toggleShareForm(documentId) {
    setOpenShareFormId((prev) => (prev === documentId ? null : documentId))
    setShareStatus(null)
    setShareEmail('')
  }

  async function handleShare(documentId) {
    if (!shareEmail.trim() || sharing) return

    setSharing(true)
    setShareStatus({ ok: null, message: 'Sharing...' })

    try {
      const res = await authedFetch(`${API_URL}/documents/${documentId}/share`, (token) => ({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ email: shareEmail.trim(), permission: sharePermission }),
      }))

      if (res.status === 401) {
        handleAuthFailure()
        return
      }
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.detail || `Share failed (${res.status})`)
      }

      const data = await res.json()
      setShareStatus({ ok: true, message: `Shared with ${data.shared_with} (${data.permission}).` })
      setShareEmail('')
      loadShares(documentId)
    } catch (err) {
      if (err.message === 'SESSION_EXPIRED') {
        handleAuthFailure()
      } else {
        setShareStatus({ ok: false, message: err.message })
      }
    } finally {
      setSharing(false)
    }
  }

  async function handleRevoke(documentId, targetEmail) {
    try {
      const res = await authedFetch(
        `${API_URL}/documents/${documentId}/share/${encodeURIComponent(targetEmail)}`,
        (token) => ({
          method: 'DELETE',
          headers: { Authorization: `Bearer ${token}` },
        })
      )

      if (res.status === 401) {
        handleAuthFailure()
        return
      }
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.detail || `Revoke failed (${res.status})`)
      }

      loadShares(documentId)
    } catch (err) {
      if (err.message === 'SESSION_EXPIRED') {
        handleAuthFailure()
      } else {
        setShareStatus({ ok: false, message: err.message })
      }
    }
  }

  if (loading) {
    return <p className="documents-loading">Loading documents...</p>
  }

  const filteredDocuments = documents.filter((doc) =>
    doc.title.toLowerCase().includes(searchQuery.trim().toLowerCase())
  )

  return (
    <section className="document-manager">
      <h2>Your documents</h2>
      {error && <p className="status-error">{error}</p>}

      {documents.length > 5 && (
        <input
          type="text"
          className="document-search"
          placeholder="Search documents..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      )}

      {documents.length === 0 && !error && (
        <p className="empty-state">No documents uploaded yet.</p>
      )}

      {documents.length > 0 && filteredDocuments.length === 0 && (
        <p className="empty-state">No documents match "{searchQuery}".</p>
      )}

      <div className="document-grid">
        {filteredDocuments.map((doc) => (
          <details key={doc.id} className="document-card" data-permission={doc.permission}>
            <summary className="document-card-summary">
              <span className="document-title">{doc.title}</span>
              <span className="document-permission-badge" data-permission={doc.permission}>
                {doc.permission}
              </span>
            </summary>

            <div className="document-card-body">
              {doc.permission === 'owner' && (
                <>
                  <div className="shares-list">
                    {(shares[doc.id] || []).length === 0 ? (
                      <p className="empty-state shares-empty">Not shared with anyone yet.</p>
                    ) : (
                      <>
                        <span className="shares-label">Shared with</span>
                        <div className="share-chip-row">
                          {(shares[doc.id] || []).map((share) => (
                            <div key={share.email} className="share-row">
                              <span>{share.email}</span>
                              <span className="document-permission-badge" data-permission={share.permission}>
                                {share.permission}
                              </span>
                              <button onClick={() => handleRevoke(doc.id, share.email)}>
                                Revoke
                              </button>
                            </div>
                          ))}
                        </div>
                      </>
                    )}
                  </div>

                  <button
                    className="share-toggle-button"
                    onClick={() => toggleShareForm(doc.id)}
                  >
                    {openShareFormId === doc.id ? 'Cancel' : 'Share with someone'}
                  </button>
                </>
              )}

              {doc.permission !== 'owner' && (
                <p className="empty-state">Only the owner can manage sharing.</p>
              )}

              {openShareFormId === doc.id && (
                <div className="share-form">
                  <input
                    type="email"
                    placeholder="Email to share with"
                    value={shareEmail}
                    onChange={(e) => setShareEmail(e.target.value)}
                    disabled={sharing}
                  />
                  <select
                    value={sharePermission}
                    onChange={(e) => setSharePermission(e.target.value)}
                    disabled={sharing}
                  >
                    <option value="viewer">Viewer</option>
                    <option value="owner">Owner</option>
                  </select>
                  <button
                    onClick={() => handleShare(doc.id)}
                    disabled={sharing || !shareEmail.trim()}
                  >
                    {sharing ? 'Sharing...' : 'Share'}
                  </button>

                  {shareStatus && (
                    <p className={shareStatus.ok === false ? 'status-error' : 'status-ok'}>
                      {shareStatus.message}
                    </p>
                  )}
                </div>
              )}
            </div>
          </details>
        ))}
      </div>
    </section>
  )
}

export default DocumentManager