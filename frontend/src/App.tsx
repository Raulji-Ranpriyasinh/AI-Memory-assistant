import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { useAuthStore } from './store/authStore';
import { RTL_LOCALES } from './i18n/i18n';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardLayout from './components/layout/DashboardLayout';
import DashboardPage from './pages/DashboardPage';
import MealsPage from './pages/MealsPage';
import AssistantPage from './pages/AssistantPage';
import MoodPage from './pages/MoodPage';
import ProfilePage from './pages/ProfilePage';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
}

function App() {
  const initAuth = useAuthStore((s) => s.init);
  const { i18n } = useTranslation();
  const [toasterPosition, setToasterPosition] = useState<'bottom-right' | 'bottom-left'>('bottom-right');

  useEffect(() => {
    initAuth();
  }, [initAuth]);

  // Update toast position when locale changes
  useEffect(() => {
    const isRTL = RTL_LOCALES.includes(i18n.language);
    setToasterPosition(isRTL ? 'bottom-left' : 'bottom-right');
  }, [i18n.language]);

  return (
    <BrowserRouter>
      <Toaster
        position={toasterPosition}
        toastOptions={{
          style: {
            borderRadius: '12px',
            fontFamily: 'inherit',
            fontSize: '14px',
            boxShadow: '0 10px 25px -5px rgba(0,0,0,0.15)',
          },
          duration: 3500,
        }}
      />
      <Routes>
        {/* Standalone pages (no layout) */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />

        {/* Dashboard pages (with layout) */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <DashboardLayout>
                <DashboardPage />
              </DashboardLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/meals"
          element={
            <ProtectedRoute>
              <DashboardLayout>
                <MealsPage />
              </DashboardLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/assistant"
          element={
            <ProtectedRoute>
              <DashboardLayout>
                <AssistantPage />
              </DashboardLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/mood"
          element={
            <ProtectedRoute>
              <DashboardLayout>
                <MoodPage />
              </DashboardLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/profile"
          element={
            <ProtectedRoute>
              <DashboardLayout>
                <ProfilePage />
              </DashboardLayout>
            </ProtectedRoute>
          }
        />

        {/* Redirects */}
        <Route path="*" element={<Navigate to="/dashboard" />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
