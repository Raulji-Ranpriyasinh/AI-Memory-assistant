import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import api from '../api/client';
import toast from 'react-hot-toast';
import { Heart, Calendar, Clock } from 'lucide-react';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const EMOTIONS = [
  { value: 'happy', emoji: '😊', label: 'mood.emotions.happy', color: 'from-yellow-400 to-amber-400' },
  { value: 'calm', emoji: '😌', label: 'mood.emotions.calm', color: 'from-blue-400 to-cyan-400' },
  { value: 'energised', emoji: '⚡', label: 'mood.emotions.energised', color: 'from-orange-400 to-yellow-400' },
  { value: 'tired', emoji: '😴', label: 'mood.emotions.tired', color: 'from-purple-400 to-indigo-400' },
  { value: 'anxious', emoji: '😰', label: 'mood.emotions.anxious', color: 'from-red-400 to-pink-400' },
  { value: 'sad', emoji: '😢', label: 'mood.emotions.sad', color: 'from-blue-500 to-indigo-500' },
  { value: 'irritable', emoji: '😤', label: 'mood.emotions.irritable', color: 'from-orange-500 to-red-500' },
  { value: 'unwell', emoji: '🤒', label: 'mood.emotions.unwell', color: 'from-red-500 to-rose-500' },
  { value: 'other', emoji: '💭', label: 'mood.emotions.other', color: 'from-slate-400 to-slate-500' },
];

export default function MoodPage() {
  const { t: tm } = useTranslation('mood');
  const [selectedEmotion, setSelectedEmotion] = useState('');
  const [stressLevel, setStressLevel] = useState(5);
  const [notes, setNotes] = useState('');
  const [moodHistory, setMoodHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);

  useEffect(() => {
    loadMoodHistory();
  }, []);

  const loadMoodHistory = async () => {
    setHistoryLoading(true);
    try {
      const res = await api.get('/mood/history');
      setMoodHistory(res.data.data || []);
    } catch {
      // Ignore errors
    } finally {
      setHistoryLoading(false);
    }
  };

  const handleLogMood = async () => {
    if (!selectedEmotion) {
      toast.error(tm('selectEmotion'));
      return;
    }

    setLoading(true);
    try {
      const res = await api.post('/mood', {
        emotion: selectedEmotion,
        stressLevel,
        notes,
        timestamp: new Date().toISOString(),
      });
      toast.success(tm('moodLogged'));
      setSelectedEmotion('');
      setStressLevel(5);
      setNotes('');
      loadMoodHistory();
      if (res.data.data?.aiHint) {
        toast.success(`${tm('aiInsight')} ${res.data.data.aiHint}`);
      }
    } catch {
      toast.error(tm('logFailed'));
    } finally {
      setLoading(false);
    }
  };

  // Group by date
  const groupedMoods = moodHistory.reduce((acc: Record<string, any[]>, mood) => {
    const date = new Date(mood.timestamp || mood.createdAt).toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
    if (!acc[date]) acc[date] = [];
    acc[date].push(mood);
    return acc;
  }, {});

  const getEmotionInfo = (value: string) => {
    return EMOTIONS.find((e) => e.value === value) || EMOTIONS[EMOTIONS.length - 1];
  };

  const getStressColor = (level: number) => {
    if (level <= 3) return 'text-emerald-600 bg-emerald-100';
    if (level <= 6) return 'text-amber-600 bg-amber-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <div className="p-4 md:p-6 space-y-6 max-w-4xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-slate-800">
          <span className="bg-gradient-to-r from-pink-500 to-rose-500 bg-clip-text text-transparent">
            Mood Tracker
          </span>
        </h1>
        <p className="text-slate-500 mt-1">Track your emotional well-being</p>
      </div>

      {/* Log Mood Section */}
      <div className="card p-5">
        <h2 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
          <Heart className="w-5 h-5 text-pink-500" />
          {tm('title')}
        </h2>

        {/* Emotion Picker */}
        <div className="mb-4">
          <label className="block text-sm font-semibold text-slate-700 mb-3">
            How are you feeling?
          </label>
          <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-9 gap-2">
            {EMOTIONS.map((e) => (
              <button
                key={e.value}
                onClick={() => setSelectedEmotion(e.value)}
                className={`p-3 rounded-xl text-center transition-all duration-200 min-h-[80px] flex flex-col items-center justify-center gap-1 ${
                  selectedEmotion === e.value
                    ? `bg-gradient-to-br ${e.color} text-white shadow-md scale-105`
                    : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
                }`}
                aria-pressed={selectedEmotion === e.value}
              >
                <span className="text-2xl">{e.emoji}</span>
                <span className={`text-[10px] font-medium ${selectedEmotion === e.value ? 'text-white/90' : 'text-slate-500'}`}>
                  {tm(e.label)}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Stress Level Slider */}
        <div className="mb-4">
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Stress level: <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${getStressColor(stressLevel)}`}>{stressLevel}/10</span>
          </label>
          <input
            type="range"
            min="1"
            max="10"
            value={stressLevel}
            onChange={(e) => setStressLevel(parseInt(e.target.value))}
            className="w-full h-2 bg-gradient-to-r from-emerald-400 via-amber-400 to-red-500 rounded-full appearance-none cursor-pointer"
            role="slider"
            aria-valuemin={1}
            aria-valuemax={10}
            aria-valuenow={stressLevel}
          />
          <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>No stress</span>
            <span>Extreme</span>
          </div>
        </div>

        {/* Notes */}
        <div className="mb-4">
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Notes (optional)
          </label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder={tm('notesPlaceholder')}
            className="w-full px-4 py-3 border-2 border-slate-200 rounded-xl focus:ring-2 focus:ring-pink-500 focus:border-pink-500 outline-none transition-all duration-200 resize-none"
            rows={3}
            dir="auto"
          />
        </div>

        {/* Submit */}
        <button
          onClick={handleLogMood}
          disabled={loading}
          className="w-full bg-gradient-to-r from-pink-500 to-rose-500 text-white py-3 rounded-xl hover:from-pink-600 hover:to-rose-600 disabled:opacity-50 disabled:cursor-not-allowed font-semibold shadow-md hover:shadow-lg transition-all duration-200 min-h-[44px]"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <LoadingSpinner size="sm" />
              Logging mood...
            </span>
          ) : (
            tm('logMood')
          )}
        </button>
      </div>

      {/* Mood History */}
      <div className="card p-5">
        <h2 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
          <Calendar className="w-5 h-5 text-pink-500" />
          Mood History
        </h2>

        {historyLoading ? (
          <div className="flex justify-center py-8">
            <LoadingSpinner />
          </div>
        ) : Object.keys(groupedMoods).length === 0 ? (
          <div className="text-center py-8">
            <Heart className="w-12 h-12 text-slate-400 mx-auto mb-3" />
            <p className="text-slate-500 font-medium">No mood entries yet</p>
            <p className="text-slate-400 text-sm">Start tracking how you feel</p>
          </div>
        ) : (
          <div className="space-y-6">
            {Object.entries(groupedMoods).map(([date, moods]) => (
              <div key={date}>
                <h3 className="text-sm font-semibold text-slate-500 mb-3 pb-2 border-b border-slate-200">
                  {date}
                </h3>
                <div className="space-y-3">
                  {moods.map((mood: any, i: number) => {
                    const emotion = getEmotionInfo(mood.emotion);
                    return (
                      <div
                        key={i}
                        className="bg-gradient-to-r from-slate-50 to-white rounded-xl border border-slate-200 p-4 flex items-start gap-3"
                      >
                        <div className={`w-12 h-12 bg-gradient-to-br ${emotion.color} rounded-xl flex items-center justify-center text-2xl flex-shrink-0 shadow-sm`}>
                          {emotion.emoji}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-sm font-semibold text-slate-700 capitalize">
                              {tm(`emotions.${mood.emotion}`)}
                            </span>
                            <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${getStressColor(mood.stressLevel)}`}>
                              Stress: {mood.stressLevel}/10
                            </span>
                          </div>
                          {mood.notes && (
                            <p className="text-sm text-slate-600 line-clamp-2">{mood.notes}</p>
                          )}
                          <div className="flex items-center gap-1 mt-1.5 text-slate-400">
                            <Clock className="w-3.5 h-3.5" />
                            <span className="text-xs">
                              {new Date(mood.timestamp || mood.createdAt).toLocaleTimeString([], {
                                hour: '2-digit',
                                minute: '2-digit',
                              })}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
