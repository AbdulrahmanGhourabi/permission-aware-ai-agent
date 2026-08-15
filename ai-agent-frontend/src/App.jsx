import { useState, useEffect } from 'react'
import Login from './components/Login'
import Chat from './components/Chat'
import DocumentManager from './components/DocumentManager'
import ErrorBoundary from './components/ErrorBoundary'
import { supabase } from './supabaseClient'
import './styles/App.css'
import './styles/Chat.css'
import './styles/DocumentManager.css'

function App() {
  const [user, setUser] = useState(null)
  const [checkingSession, setCheckingSession] = useState(true)

  useEffect(() => {
    // Resolve the current session once on load, so we don't flash the
    // Login screen before Login.jsx's own check finishes.
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null)
      setCheckingSession(false)
    })

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
    })

    return () => listener.subscription.unsubscribe()
  }, [])

  const handleLogout = async () => {
    await supabase.auth.signOut()
    setUser(null)
  }

  if (checkingSession) {
    return <p className="login-loading">Loading...</p>
  }

  if (!user) {
    return <Login onLogin={setUser} />
  }

  return (
    <ErrorBoundary>
      <div>
        <header>
          <h1>Enterprise AI Knowledge Assistant</h1>
          <p>Signed in as {user.email}</p>
          <button onClick={handleLogout}>Sign out</button>
        </header>

        <main>
          <DocumentManager onSessionExpired={handleLogout} />
          <Chat onSessionExpired={handleLogout} />
        </main>
      </div>
    </ErrorBoundary>
  )
}

export default App