import { useNavigate, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '../../store/authStore';
import {
  LayoutDashboard,
  UtensilsCrossed,
  MessageSquare,
  Heart,
  User,
  LogOut,
  X,
} from 'lucide-react';

interface SidebarProps {
  collapsed?: boolean;
  onToggle?: () => void;
}

const navItems = [
  { path: '/dashboard', icon: LayoutDashboard, label: 'sidebar.dashboard' },
  { path: '/meals', icon: UtensilsCrossed, label: 'sidebar.meals' },
  { path: '/assistant', icon: MessageSquare, label: 'sidebar.assistant' },
  { path: '/mood', icon: Heart, label: 'sidebar.mood' },
  { path: '/profile', icon: User, label: 'sidebar.profile' },
];

export default function Sidebar({ collapsed = false, onToggle }: SidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useTranslation('common');
  const user = useAuthStore((s) => s.user);
  const clearAuth = useAuthStore((s) => s.clearAuth);

  const handleLogout = () => {
    clearAuth();
    navigate('/login');
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <aside
      className={`flex flex-col h-full bg-sidebar text-white transition-all duration-300 ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Logo / Header */}
      <div className="flex items-center justify-between px-4 py-6 border-b border-white/10">
        {!collapsed && (
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-lg">D</span>
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
                Delight Health
              </h1>
            </div>
          </div>
        )}
        {collapsed && (
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-xl flex items-center justify-center shadow-lg mx-auto">
            <span className="text-white font-bold text-lg">D</span>
          </div>
        )}
        {onToggle && (
          <button
            onClick={onToggle}
            className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
            aria-label="Toggle sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.path);
          return (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium transition-all duration-200 min-h-[44px] ${
                active
                  ? 'bg-gradient-to-r from-blue-600/40 to-cyan-500/20 text-white shadow-md border-l-4 border-cyan-400'
                  : 'text-slate-300 hover:bg-white/5 hover:text-white'
              }`}
              title={collapsed ? t(item.label) : undefined}
            >
              <Icon className={`w-5 h-5 flex-shrink-0 ${active ? 'text-cyan-400' : ''}`} />
              {!collapsed && <span>{t(item.label)}</span>}
            </button>
          );
        })}
      </nav>

      {/* User / Footer */}
      <div className="px-3 py-4 border-t border-white/10">
        {!collapsed && (
          <div className="mb-3 px-3">
            <p className="text-sm font-medium text-white truncate">
              {user?.profile?.firstName || user?.email?.split('@')[0] || 'User'}
            </p>
            <p className="text-xs text-slate-400 truncate">{user?.email}</p>
          </div>
        )}
        <button
          onClick={handleLogout}
          className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium text-slate-300 hover:bg-red-500/20 hover:text-red-400 transition-all duration-200 min-h-[44px] ${
            collapsed ? 'justify-center' : ''
          }`}
          title={collapsed ? t('logout') : undefined}
        >
          <LogOut className="w-5 h-5 flex-shrink-0" />
          {!collapsed && <span>{t('logout')}</span>}
        </button>
      </div>
    </aside>
  );
}
