import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '../store/authStore';
import api from '../api/client';
import toast from 'react-hot-toast';
import GlucoseChart from '../components/ui/GlucoseChart';
import CGMSummaryCard from '../components/ui/CGMSummaryCard';
import { Plus, Activity, TrendingUp, TrendingDown, Zap, AlertCircle } from 'lucide-react';
import LoadingSpinner from '../components/ui/LoadingSpinner';

export default function DashboardPage() {
  const { t: tg } = useTranslation('cgm');
  const { t: tf } = useTranslation('food');
  const { t: tm } = useTranslation('mood');
  const user = useAuthStore((s) => s.user);
  const [cgmSummary, setCgmSummary] = useState<any>(null);
  const [cgmHistory, setCgmHistory] = useState<any[]>([]);
  const [manualGlucose, setManualGlucose] = useState('');
  const [recentMeals, setRecentMeals] = useState<any[]>([]);
  const [recentMoods, setRecentMoods] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [summaryRes, historyRes, mealsRes, moodsRes] = await Promise.allSettled([
        api.get('/cgm/summary?period=24h'),
        api.get('/cgm/history'),
        api.get('/food/history?limit=3'),
        api.get('/mood/history?limit=3'),
      ]);

      if (summaryRes.status === 'fulfilled') setCgmSummary(summaryRes.value.data.data);
      if (historyRes.status === 'fulfilled') setCgmHistory(historyRes.value.data.data);
      if (mealsRes.status === 'fulfilled') setRecentMeals(mealsRes.value.data.data || []);
      if (moodsRes.status === 'fulfilled') setRecentMoods(moodsRes.value.data.data || []);
    } catch {
      // Ignore errors
    } finally {
      setLoading(false);
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
      loadDashboardData();
    } catch {
      toast.error(tg('saveFailed'));
    }
  };

  const getGlucoseColor = (value: number) => {
    if (value >= 70 && value <= 140) return 'text-emerald-600';
    if (value > 140 && value <= 180) return 'text-amber-600';
    return 'text-red-600';
  };

  const getGlucoseBg = (value: number) => {
    if (value >= 70 && value <= 140) return 'bg-emerald-100 border-emerald-200';
    if (value > 140 && value <= 180) return 'bg-amber-100 border-amber-200';
    return 'bg-red-100 border-red-200';
  };

  const latestGlucose = cgmHistory.length > 0 ? cgmHistory[0].glucoseMgDl : null;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  const greeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };

  return (
    <div className="p-4 md:p-6 space-y-6 max-w-7xl mx-auto">
      {/* Welcome Header */}
      <div className="mb-2">
        <h1 className="text-2xl md:text-3xl font-bold text-slate-800">
          {greeting()},{' '}
          <span className="bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
            {user?.profile?.firstName || user?.email?.split('@')[0] || 'there'}!
          </span>
        </h1>
        <p className="text-slate-500 mt-1">Here's your health overview for today</p>
      </div>

      {/* Hero Glucose Card */}
      {latestGlucose !== null ? (
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-500 p-6 text-white shadow-xl">
          <div className="absolute top-0 right-0 w-40 h-40 bg-white/10 rounded-full -translate-y-10 translate-x-10" />
          <div className="relative z-10 flex items-center justify-between">
            <div>
              <p className="text-blue-100 text-sm font-medium mb-1">Latest Glucose</p>
              <p className={`text-5xl font-bold ${getGlucoseColor(latestGlucose) !== 'text-red-600' ? 'text-white' : 'text-red-200'}`}>
                {latestGlucose}
              </p>
              <p className="text-blue-100 text-sm mt-1">mg/dL</p>
            </div>
            <div className={`px-4 py-2 rounded-xl ${getGlucoseBg(latestGlucose)} bg-opacity-30`}>
              <Activity className="w-8 h-8 text-white" />
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border-2 border-dashed border-slate-300 p-8 text-center">
          <Activity className="w-12 h-12 text-slate-400 mx-auto mb-3" />
          <p className="text-slate-500 font-medium">No glucose data yet</p>
          <p className="text-slate-400 text-sm">Add your first reading below</p>
        </div>
      )}

      {/* Quick Stats */}
      {cgmSummary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="card p-4 bg-gradient-to-br from-emerald-500 to-teal-400 text-white border-0">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="w-4 h-4" />
              <span className="text-xs font-medium text-emerald-100">Average</span>
            </div>
            <p className="text-2xl font-bold">{cgmSummary.avgGlucose || 0}</p>
            <p className="text-xs text-emerald-100">mg/dL</p>
          </div>
          <div className="card p-4 bg-gradient-to-br from-blue-500 to-indigo-400 text-white border-0">
            <div className="flex items-center gap-2 mb-1">
              <Zap className="w-4 h-4" />
              <span className="text-xs font-medium text-blue-100">Time in Range</span>
            </div>
            <p className="text-2xl font-bold">{cgmSummary.timeInRangePct || 0}%</p>
            <p className="text-xs text-blue-100">70-140 mg/dL</p>
          </div>
          <div className="card p-4 bg-gradient-to-br from-amber-500 to-orange-400 text-white border-0">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="w-4 h-4" />
              <span className="text-xs font-medium text-amber-100">Spikes</span>
            </div>
            <p className="text-2xl font-bold">{cgmSummary.spikeCount || 0}</p>
            <p className="text-xs text-amber-100">last 24h</p>
          </div>
          <div className="card p-4 bg-gradient-to-br from-rose-500 to-pink-400 text-white border-0">
            <div className="flex items-center gap-2 mb-1">
              <TrendingDown className="w-4 h-4" />
              <span className="text-xs font-medium text-rose-100">Hypos</span>
            </div>
            <p className="text-2xl font-bold">{cgmSummary.hypoCount || 0}</p>
            <p className="text-xs text-rose-100">last 24h</p>
          </div>
        </div>
      )}

      {/* CGM Summary + Chart */}
      {cgmSummary && (
        <div className="card p-5">
          <h2 className="text-lg font-bold text-slate-800 mb-4">{tg('stats.avg')}</h2>
          <CGMSummaryCard
            avgGlucose={cgmSummary.avgGlucose || 0}
            minGlucose={cgmSummary.minGlucose || 0}
            maxGlucose={cgmSummary.maxGlucose || 0}
            timeInRangePct={cgmSummary.timeInRangePct || 0}
            spikeCount={cgmSummary.spikeCount || 0}
            hypoCount={cgmSummary.hypoCount || 0}
          />
          <div className="mt-4">
            <GlucoseChart data={cgmHistory} />
          </div>
        </div>
      )}

      {/* Manual Glucose Entry */}
      <div className="card p-5">
        <h3 className="text-base font-semibold text-slate-800 mb-3">{tg('manualEntry.title', 'Add Reading')}</h3>
        <div className="flex gap-2">
          <input
            type="number"
            value={manualGlucose}
            onChange={(e) => setManualGlucose(e.target.value)}
            placeholder={tg('manualEntry.placeholder', { unit: 'mg/dL' })}
            className="flex-1 px-4 py-3 border-2 border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200"
            onKeyDown={(e) => e.key === 'Enter' && handleAddGlucose()}
            inputMode="decimal"
          />
          <button
            onClick={handleAddGlucose}
            className="bg-gradient-to-r from-blue-600 to-cyan-600 text-white px-5 py-3 rounded-xl hover:from-blue-700 hover:to-cyan-700 flex items-center gap-2 min-h-[44px] font-semibold shadow-md hover:shadow-lg transition-all duration-200"
          >
            <Plus className="w-5 h-5" />
            {tg('manualEntry.add')}
          </button>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="card p-5">
        <h3 className="text-base font-semibold text-slate-800 mb-4">Recent Activity</h3>
        {(recentMeals.length === 0 && recentMoods.length === 0) ? (
          <div className="text-center py-6">
            <AlertCircle className="w-8 h-8 text-slate-400 mx-auto mb-2" />
            <p className="text-slate-500 text-sm">No recent activity</p>
          </div>
        ) : (
          <div className="space-y-3">
            {recentMeals.map((meal: any, i: number) => (
              <div key={`meal-${i}`} className="flex items-center gap-3 p-3 bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl border border-amber-100">
                <div className="w-10 h-10 bg-gradient-to-br from-amber-400 to-orange-400 rounded-xl flex items-center justify-center flex-shrink-0">
                  <span className="text-white text-sm">🍽️</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-slate-700 capitalize">{tf(`meal.${meal.mealType}`)}</p>
                  <p className="text-xs text-slate-500 truncate">{(meal.items || []).join(', ')}</p>
                </div>
                <span className="text-xs text-slate-400 flex-shrink-0">
                  {new Date(meal.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            ))}
            {recentMoods.map((mood: any, i: number) => {
              const emotionEmojis: Record<string, string> = {
                happy: '😊', calm: '😌', energised: '⚡', tired: '😴',
                anxious: '😰', sad: '😢', irritable: '😤', unwell: '🤒', other: '💭',
              };
              return (
                <div key={`mood-${i}`} className="flex items-center gap-3 p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-purple-100">
                  <div className="w-10 h-10 bg-gradient-to-br from-purple-400 to-pink-400 rounded-xl flex items-center justify-center flex-shrink-0 text-lg">
                    {emotionEmojis[mood.emotion] || '💭'}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-slate-700 capitalize">{tm(`emotions.${mood.emotion}`)}</p>
                    <p className="text-xs text-slate-500">Stress: {mood.stressLevel}/10</p>
                  </div>
                  <span className="text-xs text-slate-400 flex-shrink-0">
                    {new Date(mood.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
