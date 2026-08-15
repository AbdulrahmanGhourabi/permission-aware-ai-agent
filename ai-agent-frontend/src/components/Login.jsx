import { useState, useEffect } from 'react'
import { supabase } from '../supabaseClient'

export default function Login({ onLogin }) {
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Check if user is already logged in when the page loads
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) {
        onLogin(session.user)
      }
      setLoading(false)
    })

    // Listen for login/logout events
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session) {
        onLogin(session.user)
      }
    })

    return () => listener.subscription.unsubscribe()
  }, [onLogin])

  const handleGoogleLogin = async () => {
    await supabase.auth.signInWithOAuth({
      provider: 'google',
    })
  }

  if (loading) {
    return <p className="login-loading">Loading...</p>
  }

  return (
    <div className="login-screen">
      <div className="login-card">
        <div className="login-mark">S</div>
        <h1>Sentrion</h1>
        <p className="login-subtitle">Ask questions. Get answers only from what you're allowed to see.</p>
        <button className="login-button" onClick={handleGoogleLogin}>
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
            <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.9c1.7-1.57 2.7-3.88 2.7-6.62Z"/>
            <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.8.54-1.84.86-3.06.86-2.35 0-4.34-1.59-5.05-3.72H.9v2.33A9 9 0 0 0 9 18Z"/>
            <path fill="#FBBC05" d="M3.95 10.7A5.4 5.4 0 0 1 3.66 9c0-.59.1-1.17.29-1.7V4.97H.9A9 9 0 0 0 0 9c0 1.45.35 2.83.9 4.03l3.05-2.33Z"/>
            <path fill="#EA4335" d="M9 3.58c1.32 0 2.51.46 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0A9 9 0 0 0 .9 4.97l3.05 2.33C4.66 5.17 6.65 3.58 9 3.58Z"/>
          </svg>
          Sign in with Google
        </button>
      </div>
    </div>
  )
}