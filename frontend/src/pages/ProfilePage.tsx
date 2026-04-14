import { useAuthStore } from '../store/authStore';
import { User, Mail, Settings, Globe, Shield, Bell, Trash2 } from 'lucide-react';
import LanguageSwitcher from '../components/ui/LanguageSwitcher';
import api from '../api/client';
import toast from 'react-hot-toast';
import { useState } from 'react';

export default function ProfilePage() {
  const user = useAuthStore((s) => s.user);
  const clearAuth = useAuthStore((s) => s.clearAuth);
  const [deleting, setDeleting] = useState(false);

  const handleDeleteAccount = async () => {
    if (!confirm('Are you sure you want to delete your account? This cannot be undone.')) return;
    setDeleting(true);
    try {
      await api.delete('/auth/me');
      clearAuth();
      toast.success('Account deleted successfully');
    } catch {
      toast.error('Failed to delete account');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="p-4 md:p-6 space-y-6 max-w-3xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-slate-800">
          <span className="bg-gradient-to-r from-indigo-500 to-purple-500 bg-clip-text text-transparent">
            Profile & Settings
          </span>
        </h1>
        <p className="text-slate-500 mt-1">Manage your account</p>
      </div>

      {/* User Info Card */}
      <div className="card overflow-hidden">
        <div className="bg-gradient-to-r from-indigo-500 to-purple-500 h-24" />
        <div className="px-5 pb-5 -mt-12">
          <div className="flex items-end gap-4">
            <div className="w-20 h-20 bg-white rounded-2xl shadow-lg flex items-center justify-center border-4 border-white">
              <div className="w-16 h-16 bg-gradient-to-br from-indigo-400 to-purple-400 rounded-xl flex items-center justify-center">
                <User className="w-8 h-8 text-white" />
              </div>
            </div>
            <div className="pb-1">
              <h2 className="text-xl font-bold text-slate-800">
                {user?.profile?.firstName} {user?.profile?.lastName}
              </h2>
              <p className="text-sm text-slate-500">{user?.email}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Account Details */}
      <div className="card p-5">
        <h3 className="text-base font-semibold text-slate-800 mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5 text-indigo-500" />
          Account Details
        </h3>
        <div className="space-y-4">
          <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-xl">
            <User className="w-5 h-5 text-slate-400 flex-shrink-0" />
            <div>
              <p className="text-xs text-slate-500 font-medium">Name</p>
              <p className="text-sm font-semibold text-slate-700">
                {user?.profile?.firstName || '—'} {user?.profile?.lastName || ''}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-xl">
            <Mail className="w-5 h-5 text-slate-400 flex-shrink-0" />
            <div>
              <p className="text-xs text-slate-500 font-medium">Email</p>
              <p className="text-sm font-semibold text-slate-700">{user?.email}</p>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-xl">
            <Settings className="w-5 h-5 text-slate-400 flex-shrink-0" />
            <div>
              <p className="text-xs text-slate-500 font-medium">Role</p>
              <p className="text-sm font-semibold text-slate-700 capitalize">{user?.role || 'user'}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Language Settings */}
      <div className="card p-5">
        <h3 className="text-base font-semibold text-slate-800 mb-4 flex items-center gap-2">
          <Globe className="w-5 h-5 text-indigo-500" />
          Language & Region
        </h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
            <div className="flex items-center gap-3">
              <Globe className="w-5 h-5 text-slate-400" />
              <div>
                <p className="text-sm font-semibold text-slate-700">App Language</p>
                <p className="text-xs text-slate-500">Choose your preferred language</p>
              </div>
            </div>
            <LanguageSwitcher />
          </div>
        </div>
      </div>

      {/* Notifications */}
      <div className="card p-5">
        <h3 className="text-base font-semibold text-slate-800 mb-4 flex items-center gap-2">
          <Bell className="w-5 h-5 text-indigo-500" />
          Notifications
        </h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
            <div>
              <p className="text-sm font-semibold text-slate-700">Push Notifications</p>
              <p className="text-xs text-slate-500">Get reminders for meal logging and mood tracking</p>
            </div>
            <span className="text-xs font-medium text-slate-400 bg-slate-200 px-3 py-1 rounded-full">
              Coming soon
            </span>
          </div>
        </div>
      </div>

      {/* Danger Zone */}
      <div className="card p-5 border-2 border-red-200">
        <h3 className="text-base font-semibold text-red-600 mb-4 flex items-center gap-2">
          <Trash2 className="w-5 h-5" />
          Danger Zone
        </h3>
        <div className="flex items-center justify-between p-3 bg-red-50 rounded-xl">
          <div>
            <p className="text-sm font-semibold text-red-700">Delete Account</p>
            <p className="text-xs text-red-600">Permanently delete your account and all data</p>
          </div>
          <button
            onClick={handleDeleteAccount}
            disabled={deleting}
            className="bg-red-600 text-white px-4 py-2 rounded-xl hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-semibold min-h-[44px] transition-colors"
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  );
}
