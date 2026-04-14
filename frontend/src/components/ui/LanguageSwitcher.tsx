import { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Globe, Check, ChevronDown } from 'lucide-react';
import { SUPPORTED_LOCALES, RTL_LOCALES } from '../../i18n/i18n';
import type { LocaleInfo } from '../../i18n/i18n';

export default function LanguageSwitcher() {
  const { i18n, t } = useTranslation('common');
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const currentLocale = SUPPORTED_LOCALES.find((l: LocaleInfo) => l.code === i18n.language) ?? SUPPORTED_LOCALES[0];
  const isRTL = RTL_LOCALES.includes(i18n.language);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close on Escape key
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleChangeLocale = (locale: LocaleInfo) => {
    i18n.changeLanguage(locale.code);
    setOpen(false);
  };

  return (
    <div className="relative" ref={containerRef}>
      <button
        id="language-switcher-btn"
        onClick={() => setOpen((prev) => !prev)}
        className="flex items-center gap-1.5 px-3 py-2 rounded-lg hover:bg-white/10 transition-all duration-200 text-sm font-medium min-h-[44px] min-w-[44px]"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={t('selectLanguage')}
      >
        <Globe className="w-4 h-4 shrink-0" aria-hidden="true" />
        <span className="hidden sm:inline max-w-[80px] truncate" lang={currentLocale.locale}>
          {currentLocale.label}
        </span>
        <ChevronDown
          className={`w-3 h-3 shrink-0 transition-transform duration-200 hidden sm:block ${open ? 'rotate-180' : ''}`}
          aria-hidden="true"
        />
      </button>

      {/* Dropdown */}
      {open && (
        <div
          className={`absolute ${isRTL ? 'left-0' : 'right-0'} top-full mt-2 w-52 bg-white rounded-xl shadow-xl border border-slate-100 py-1.5 z-[200] animate-fade-in`}
          role="listbox"
          aria-label={t('selectLanguage')}
        >
          {/* Grouped: LTR languages then RTL */}
          <div className="px-2 pb-1">
            <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest px-2 py-1">
              Language / Langue / 语言
            </p>
          </div>
          {SUPPORTED_LOCALES.map((locale: LocaleInfo) => {
            const isActive = i18n.language === locale.code;
            return (
              <button
                key={locale.code}
                onClick={() => handleChangeLocale(locale)}
                className={`w-full px-4 py-2.5 text-sm flex items-center justify-between gap-2 transition-colors duration-150 min-h-[44px] ${
                  isActive
                    ? 'bg-blue-50 text-blue-700 font-semibold'
                    : 'text-slate-700 hover:bg-slate-50'
                }`}
                role="option"
                aria-selected={isActive}
                dir={locale.dir}
                lang={locale.locale}
              >
                <span>{locale.label}</span>
                {isActive && (
                  <Check className="w-4 h-4 text-blue-600 shrink-0" aria-hidden="true" />
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
