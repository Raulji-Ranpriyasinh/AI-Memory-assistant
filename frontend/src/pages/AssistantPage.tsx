import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import api from '../api/client';
import toast from 'react-hot-toast';
import { AlertTriangle, Send, Sparkles, MessageSquare, Lightbulb } from 'lucide-react';
import LoadingSpinner from '../components/ui/LoadingSpinner';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

const SUGGESTED_PROMPTS = [
  { icon: MessageSquare, text: "How am I doing today?", label: "Health summary" },
  { icon: Lightbulb, text: "What should I eat for breakfast?", label: "Meal suggestion" },
  { icon: Sparkles, text: "Analyze my glucose trends", label: "Glucose analysis" },
  { icon: MessageSquare, text: "Tips to improve my mood", label: "Wellness tips" },
];

export default function AssistantPage() {
  const { t: tc } = useTranslation('chat');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [memories, setMemories] = useState<any[]>([]);
  const [memoriesLoading, setMemoriesLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadChatHistory();
    loadMemories();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadChatHistory = async () => {
    try {
      const res = await api.get('/chat/history');
      const history = res.data.data || [];
      if (history.length > 0) {
        setMessages(history.map((m: any) => ({
          role: m.role,
          content: m.content || m.message || '',
        })));
      }
    } catch {
      // Ignore errors
    }
  };

  const loadMemories = async () => {
    setMemoriesLoading(true);
    try {
      const res = await api.get('/ai/memories');
      const data = res.data.data || res.data || [];
      // Handle various response formats
      if (Array.isArray(data)) {
        setMemories(data.slice(0, 10));
      } else if (data.memories) {
        setMemories(data.memories.slice(0, 10));
      } else if (data.longTermMemory) {
        setMemories(data.longTermMemory.slice(0, 10));
      }
    } catch {
      // Ignore errors - memories are optional
    } finally {
      setMemoriesLoading(false);
    }
  };

  const handleSendMessage = async (messageText?: string) => {
    const text = messageText || input.trim();
    if (!text) return;

    const userMessage: ChatMessage = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await api.post('/chat', { message: text });
      const aiResponse = res.data.data?.response || res.data.response || tc('aiError');
      setMessages((prev) => [...prev, { role: 'assistant', content: aiResponse }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: tc('aiError') },
      ]);
      toast.error(tc('aiError'));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex h-[calc(100vh-0px)]">
      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Medical Disclaimer */}
        <div className="bg-amber-50 border-b border-amber-200 px-4 py-2.5 flex items-start gap-2 flex-shrink-0">
          <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-xs font-semibold text-amber-800">{tc('disclaimer.heading')}</p>
            <p className="text-[10px] text-amber-700">{tc('disclaimer.body')}</p>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-2xl flex items-center justify-center mb-4 shadow-lg">
                <Sparkles className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-xl font-bold text-slate-800 mb-2">Your AI Health Assistant</h2>
              <p className="text-slate-500 max-w-sm mb-6">
                Ask me anything about your health data, nutrition, mood, and more. I remember your history and can provide personalized insights.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg">
                {SUGGESTED_PROMPTS.map((prompt, i) => (
                  <button
                    key={i}
                    onClick={() => handleSendMessage(prompt.text)}
                    className="flex items-center gap-3 p-3 bg-white rounded-xl border border-slate-200 hover:border-blue-300 hover:shadow-md transition-all duration-200 text-left"
                  >
                    <div className="w-8 h-8 bg-gradient-to-br from-blue-100 to-cyan-100 rounded-lg flex items-center justify-center flex-shrink-0">
                      <prompt.icon className="w-4 h-4 text-blue-600" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-700">{prompt.text}</p>
                      <p className="text-xs text-slate-400">{prompt.label}</p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] md:max-w-[70%] rounded-2xl px-4 py-3 ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-br-md'
                      : 'bg-white border border-slate-200 text-slate-800 rounded-bl-md shadow-sm'
                  }`}
                >
                  {msg.role === 'assistant' && (
                    <div className="flex items-center gap-2 mb-1.5">
                      <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-full flex items-center justify-center">
                        <Sparkles className="w-3.5 h-3.5 text-white" />
                      </div>
                      <span className="text-xs font-medium text-slate-500">AI Assistant</span>
                    </div>
                  )}
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
                <div className="flex items-center gap-2">
                  <LoadingSpinner size="sm" />
                  <span className="text-sm text-slate-500">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="border-t border-slate-200 p-4 bg-white flex-shrink-0">
          <div className="flex items-center gap-2 max-w-3xl mx-auto">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask me anything about your health..."
              className="flex-1 px-4 py-3 border-2 border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200"
              disabled={loading}
              dir="auto"
            />
            <button
              onClick={() => handleSendMessage()}
              disabled={!input.trim() || loading}
              className="bg-gradient-to-r from-blue-600 to-cyan-600 text-white p-3 rounded-xl hover:from-blue-700 hover:to-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-md min-h-[44px] min-w-[44px] flex items-center justify-center"
              aria-label="Send message"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Memory Panel (desktop only) */}
      <div className="hidden lg:flex flex-col w-72 border-l border-slate-200 bg-slate-50 flex-shrink-0">
        <div className="px-4 py-4 border-b border-slate-200 bg-white">
          <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-blue-600" />
            AI Memory
          </h3>
          <p className="text-xs text-slate-500 mt-1">What I know about you</p>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {memoriesLoading ? (
            <div className="flex justify-center py-8">
              <LoadingSpinner size="sm" />
            </div>
          ) : memories.length === 0 ? (
            <div className="text-center py-8">
              <div className="w-12 h-12 bg-slate-200 rounded-xl flex items-center justify-center mx-auto mb-3">
                <MessageSquare className="w-6 h-6 text-slate-400" />
              </div>
              <p className="text-slate-500 text-sm font-medium">No memories yet</p>
              <p className="text-slate-400 text-xs mt-1">Start chatting and I'll remember important details</p>
            </div>
          ) : (
            <div className="space-y-2">
              {memories.map((memory: any, i: number) => (
                <div
                  key={i}
                  className="bg-white rounded-xl p-3 border border-slate-200 shadow-sm"
                >
                  <p className="text-xs text-slate-700 line-clamp-3">
                    {memory.content || memory.text || memory.summary || JSON.stringify(memory)}
                  </p>
                  {memory.category && (
                    <span className="inline-block mt-1.5 px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-[10px] font-medium">
                      {memory.category}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
