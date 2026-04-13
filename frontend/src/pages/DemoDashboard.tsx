import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import api from '../api/client';
import toast from 'react-hot-toast';
import GlucoseChart from '../components/ui/GlucoseChart';
import CGMSummaryCard from '../components/ui/CGMSummaryCard';
import ChatWindow from '../components/ui/ChatWindow';
import LanguageSwitcher from '../components/ui/LanguageSwitcher';
import { LogOut, Plus, Trash2, AlertTriangle } from 'lucide-react';
import { RTL_LOCALES } from '../i18n/i18n';

const EMOTIONS = [
  { value: 'happy', emoji: '😊', label: 'mood.emotions.happy' },
  { value: 'calm', emoji: '😌', label: 'mood.emotions.calm' },
  { value: 'energised', emoji: '⚡', label: 'mood.emotions.energised' },
  { value: 'tired', emoji: '😴', label: 'mood.emotions.tired' },
  { value: 'anxious', emoji: '😰', label: 'mood.emotions.anxious' },
  { value: 'sad', emoji: '😢', label: 'mood.emotions.sad' },
  { value: 'irritable', emoji: '😤', label: 'mood.emotions.irritable' },
  { value: 'unwell', emoji: '🤒', label: 'mood.emotions.unwell' },
  { value: 'other', emoji: '💭', label: 'mood.emotions.other' },
];

const MEAL_TYPES = ['breakfast', 'lunch', 'dinner', 'snack'] as const;

export default function DemoDashboard() {
  const { t: tc } = useTranslation('common');
  const { t: tm } = useTranslation('mood');
  const { t: tf } = useTranslation('food');
  const { t: tg } = useTranslation('cgm');
  const { t: tc_chat } = useTranslation('chat');
  const navigate = useNavigate();
  const clearAuth = useAuthStore((s) => s.clearAuth);
  const user = useAuthStore((s) => s.user);
  const { i18n } = useTranslation();
  const isRTL = RTL_LOCALES.includes(i18n.language);

  // CGM state
  const [cgmSummary, setCgmSummary] = useState<any>(null);
  const [cgmHistory, setCgmHistory] = useState<any[]>([]);
  const [manualGlucose, setManualGlucose] = useState('');

  // Mood state
  const [selectedEmotion, setSelectedEmotion] = useState('');
  const [stressLevel, setStressLevel] = useState(5);
  const [moodNotes, setMoodNotes] = useState('');

  // Food state
  const [selectedMeal, setSelectedMeal] = useState('lunch');
  const [foodItems, setFoodItems] = useState<string[]>([]);
  const [foodInput, setFoodInput] = useState('');
  const [foodHistory, setFoodHistory] = useState<any[]>([]);

  // Chat state
  const [chatMessages, setChatMessages] = useState<any[]>([]);
  const [chatLoading, setChatLoading] = useState(false);

  // Load initial data
  useEffect(() => {
    loadCgmData();
    loadChatHistory();
  }, []);

  const loadCgmData = async () => {
    try {
      const [summaryRes, historyRes] = await Promise.all([
        api.get('/cgm/summary?period=24h'),
        api.get('/cgm/history'),
      ]);
      setCgmSummary(summaryRes.data.data);
      setCgmHistory(historyRes.data.data);
    } catch {
      // Ignore errors
    }
  };

  const loadChatHistory = async () => {
    try {
      const res = await api.get('/chat/history');
      setChatMessages(res.data.data || []);
    } catch {
      // Ignore errors
    }
  };

  const handleAddGlucose = async () => {
    const value = parseInt(manualGlucose);
    if (!value || value < 20 || value > 500) {
      toast.error(tg('saveFailed'));
      return;
    }

    try {
      await api.post('/cgm/readings', {
        readings: [
          {
            glucoseMgDl: value,
            timestamp: new Date().toISOString(),
            trend: value > 140 ? 'rising' : value < 70 ? 'falling' : 'stable',
          },
        ],
      });
      toast.success(tg('saved'));
      setManualGlucose('');
      loadCgmData();
    } catch {
      toast.error(tg('saveFailed'));
    }
  };

  const handleLogMood = async () => {
    if (!selectedEmotion) {
      toast.error(tm('selectEmotion'));
      return;
    }

    try {
      const res = await api.post('/mood', {
        emotion: selectedEmotion,
        stressLevel,
        notes: moodNotes,
        timestamp: new Date().toISOString(),
      });
      toast.success(tm('moodLogged'));
      setSelectedEmotion('');
      setMoodNotes('');
      if (res.data.data.aiHint) {
        toast.success(`${tm('aiInsight')} ${res.data.data.aiHint}`);
      }
    } catch {
      toast.error(tm('logFailed'));
    }
  };

  const handleAddFoodItem = () => {
    if (foodInput.trim()) {
      setFoodItems([...foodItems, foodInput.trim()]);
      setFoodInput('');
    }
  };

  const handleRemoveFoodItem = (index: number) => {
    setFoodItems(foodItems.filter((_, i) => i !== index));
  };

  const handleLogFood = async () => {
    if (foodItems.length === 0) {
      toast.error(tf('addFood'));
      return;
    }

    try {
      await api.post('/food/log', {
        mealType: selectedMeal,
        items: foodItems,
        timestamp: new Date().toISOString(),
      });
      toast.success(tf('mealLogged'));
      setFoodItems([]);
      setFoodHistory([...foodHistory, { mealType: selectedMeal, items: [...foodItems], timestamp: new Date() }]);
    } catch {
      toast.error(tf('logFailed'));
    }
  };

  const handleSendMessage = async (message: string) => {
    setChatMessages([...chatMessages, { role: 'user', content: message }]);
    setChatLoading(true);

    try {
      const res = await api.post('/chat', { message });
      const aiResponse = res.data.data?.response || tc_chat('aiError');
      setChatMessages((prev) => [...prev, { role: 'assistant', content: aiResponse }]);
    } catch {
      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', content: tc_chat('aiError') },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleLogout = () => {
    clearAuth();
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-slate-800 text-white px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
            <span className="text-white font-bold">D</span>
          </div>
          <div>
            <h1 className="text-lg font-bold">Delight Health</h1>
            <p className="text-xs text-slate-400">{user?.email}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <LanguageSwitcher />
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 px-3 py-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors text-sm"
            aria-label={tc('logout')}
          >
            <LogOut className="w-4 h-4" />
            <span className="hidden sm:inline">{tc('logout')}</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto p-4 space-y-6">
        {/* Panel 1: Glucose */}
        <section className="bg-white rounded-xl shadow p-4">
          <h2 className="text-lg font-bold text-slate-800 mb-4">{tg('stats.avg')}</h2>
          {cgmSummary && (
            <CGMSummaryCard
              avgGlucose={cgmSummary.avgGlucose || 0}
              minGlucose={cgmSummary.minGlucose || 0}
              maxGlucose={cgmSummary.maxGlucose || 0}
              timeInRangePct={cgmSummary.timeInRangePct || 0}
              spikeCount={cgmSummary.spikeCount || 0}
              hypoCount={cgmSummary.hypoCount || 0}
            />
          )}
          <div className="mt-4">
            <GlucoseChart data={cgmHistory} />
          </div>
          <div className="mt-4 flex gap-2">
            <input
              type="number"
              value={manualGlucose}
              onChange={(e) => setManualGlucose(e.target.value)}
              placeholder={tg('manualEntry.placeholder', { unit: 'mg/dL' })}
              className="flex-1 px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
              onKeyDown={(e) => e.key === 'Enter' && handleAddGlucose()}
              inputMode="decimal"
            />
            <button
              onClick={handleAddGlucose}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 flex items-center gap-2 min-h-[44px]"
            >
              <Plus className="w-4 h-4" />
              {tg('manualEntry.add')}
            </button>
          </div>
        </section>

        {/* Panel 2: Mood */}
        <section className="bg-white rounded-xl shadow p-4">
          <h2 className="text-lg font-bold text-slate-800 mb-4">{tm('title')}</h2>
          <div className="grid grid-cols-3 md:grid-cols-9 gap-2 mb-4">
            {EMOTIONS.map((e) => (
              <button
                key={e.value}
                onClick={() => setSelectedEmotion(e.value)}
                className={`p-3 rounded-lg text-center transition-colors min-h-[80px] ${
                  selectedEmotion === e.value
                    ? 'bg-blue-100 border-2 border-blue-600'
                    : 'bg-slate-50 hover:bg-slate-100'
                }`}
                aria-pressed={selectedEmotion === e.value}
              >
                <div className="text-2xl">{e.emoji}</div>
                <div className="text-xs text-slate-500 mt-1">{tm(e.label)}</div>
              </button>
            ))}
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              {tm('stress.label')}: {stressLevel}/10
            </label>
            <input
              type="range"
              min="1"
              max="10"
              value={stressLevel}
              onChange={(e) => setStressLevel(parseInt(e.target.value))}
              className="w-full"
              style={{
                background: `linear-gradient(to ${isRTL ? 'left' : 'right'}, #16A34A, #EAB308, #DC2626)`,
              }}
              role="slider"
              aria-valuemin={1}
              aria-valuemax={10}
              aria-valuenow={stressLevel}
              aria-valuetext={tm('stress.ariaValue', { n: stressLevel, max: 10 })}
            />
          </div>

          <textarea
            value={moodNotes}
            onChange={(e) => setMoodNotes(e.target.value)}
            placeholder={tm('notesPlaceholder')}
            className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none mb-4"
            rows={2}
            dir="auto"
          />

          <button
            onClick={handleLogMood}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 min-h-[44px]"
          >
            {tm('logMood')}
          </button>
        </section>

        {/* Panel 3: Meal */}
        <section className="bg-white rounded-xl shadow p-4">
          <h2 className="text-lg font-bold text-slate-800 mb-4">{tf('title')}</h2>
          <div className="flex gap-2 mb-4">
            {MEAL_TYPES.map((meal) => (
              <button
                key={meal}
                onClick={() => setSelectedMeal(meal)}
                className={`px-4 py-2 rounded-lg capitalize transition-colors min-h-[44px] ${
                  selectedMeal === meal
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                }`}
              >
                {tf(`meal.${meal}`)}
              </button>
            ))}
          </div>

          <div className="flex gap-2 mb-4">
            <input
              type="text"
              value={foodInput}
              onChange={(e) => setFoodInput(e.target.value)}
              placeholder={tf('manualEntry.foodInput')}
              className="flex-1 px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none min-h-[44px]"
              onKeyDown={(e) => e.key === 'Enter' && handleAddFoodItem()}
              dir="auto"
            />
          </div>

          {foodItems.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {foodItems.map((item, i) => (
                <span
                  key={i}
                  className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm flex items-center gap-1"
                >
                  {item}
                  <button onClick={() => handleRemoveFoodItem(i)} className="hover:text-blue-600 min-w-[44px] min-h-[44px] flex items-center justify-center">
                    <Trash2 className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}

          <button
            onClick={handleLogFood}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 min-h-[44px]"
          >
            {tf('logMeal')}
          </button>

          {/* Food History */}
          {foodHistory.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-medium text-slate-700 mb-2">{tf('history.title')}</h3>
              <div className="space-y-2">
                {foodHistory.map((log, i) => (
                  <div key={i} className="bg-slate-50 rounded-lg p-3 text-sm">
                    <span className="font-medium">{tf(`meal.${log.mealType}`)}</span>: {log.items.join(', ')}
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Panel 4: AI Chat */}
        <section className="bg-white rounded-xl shadow p-4">
          <h2 className="text-lg font-bold text-slate-800 mb-4">{tc_chat('title')}</h2>
          
          {/* Medical Disclaimer */}
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-3 flex items-start gap-2">
            <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-yellow-800">{tc_chat('disclaimer.heading')}</p>
              <p className="text-xs text-yellow-700 mt-1">{tc_chat('disclaimer.body')}</p>
            </div>
          </div>
          
          <ChatWindow
            messages={chatMessages.map((m) => ({
              role: m.role,
              content: m.content || m.message || '',
            }))}
            onSend={handleSendMessage}
            loading={chatLoading}
          />
        </section>
      </main>
    </div>
  );
}
