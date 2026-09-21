import { useState, useRef, useEffect, useCallback } from 'react'
import { api } from '../utils/api'

// Three openers that fit on one line — the list is a prompt, not a menu.
const SUGGESTED_QUESTIONS = [
  "What if rain doubles?",
  "Will Andheri flood?",
  "What's the worst case?",
]

function parseMarkdown(text) {
  // Simple markdown: **bold** and bullet points
  return text
    .split('\n')
    .map(line => {
      // Bullet points
      if (line.match(/^[•\-\*]\s/)) {
        return `<div class="chat-bullet">${line.replace(/^[•\-\*]\s/, '')}</div>`
      }
      // Bold text
      line = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Headers
      if (line.startsWith('##')) {
        return `<div class="chat-heading">${line.replace(/^#+\s*/, '')}</div>`
      }
      return `<div>${line || '&nbsp;'}</div>`
    })
    .join('')
}

export default function WhatIfChatbot({ timestep }) {
  const [messages, setMessages] = useState([
    {
      role: 'bot',
      content: "Ask me about flood risk anywhere in Mumbai.",
      data: null,
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(true)
  const chatEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return

    const userMsg = { role: 'user', content: text.trim() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setShowSuggestions(false)

    try {
      const result = await api.innovations.chatbotAsk(text.trim(), timestep)
      const botMsg = {
        role: 'bot',
        content: result.answer || "I couldn't process that question.",
        data: result,
      }
      setMessages(prev => [...prev, botMsg])
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'bot',
        content: "Sorry, I encountered an error processing your question. Please try again.",
        data: null,
        isError: true,
      }])
    }
    setLoading(false)
    inputRef.current?.focus()
  }, [timestep, loading])

  const handleSubmit = (e) => {
    e.preventDefault()
    sendMessage(input)
  }

  const handleSuggestion = (q) => {
    sendMessage(q)
  }

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <span className="chatbot-icon">💬</span>
        <span className="chatbot-title">What-If Simulator</span>
        <span className="chatbot-badge">AI</span>
      </div>

      <div className="chatbot-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg chat-msg-${msg.role}`}>
            {msg.role === 'bot' && (
              <div className="chat-avatar">🌊</div>
            )}
            <div className={`chat-bubble ${msg.isError ? 'chat-error' : ''}`}>
              <div
                className="chat-text"
                dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }}
              />
              {msg.data && msg.data.scenario_id && (
                <div className="chat-data-badge">
                  <span className="chat-badge-label">Scenario:</span>
                  <span className="chat-badge-value">{msg.data.scenario_id}</span>
                  {msg.data.delta !== 0 && (
                    <span className={`chat-badge-delta ${msg.data.delta > 0 ? 'danger' : 'safe'}`}>
                      {msg.data.delta > 0 ? '+' : ''}{msg.data.delta.toFixed(0)} risk
                    </span>
                  )}
                </div>
              )}
              {msg.data && msg.data.affected_cells > 0 && (
                <div className="chat-affected">
                  📍 {msg.data.affected_cells}/90 areas affected
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-msg chat-msg-bot">
            <div className="chat-avatar">🌊</div>
            <div className="chat-bubble chat-thinking">
              <div className="thinking-dots">
                <span>.</span><span>.</span><span>.</span>
              </div>
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {showSuggestions && (
        <div className="chat-suggestions">
          {SUGGESTED_QUESTIONS.map((q, i) => (
            <button key={i} className="chat-suggestion-chip" onClick={() => handleSuggestion(q)}>
              {q}
            </button>
          ))}
        </div>
      )}

      <form className="chatbot-input" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask what-if question..."
          disabled={loading}
          className="chat-input-field"
        />
        <button type="submit" disabled={loading || !input.trim()} className="chat-send-btn">
          {loading ? '⏳' : '→'}
        </button>
      </form>
    </div>
  )
}
