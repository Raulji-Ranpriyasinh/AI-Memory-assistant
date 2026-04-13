import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import common_en from './locales/en/common.json';
import auth_en from './locales/en/auth.json';
import cgm_en from './locales/en/cgm.json';
import mood_en from './locales/en/mood.json';
import food_en from './locales/en/food.json';
import chat_en from './locales/en/chat.json';

// Import all other locales dynamically will be handled via backend later
// For now we statically import all 12 locales

import common_he from './locales/he/common.json';
import auth_he from './locales/he/auth.json';
import cgm_he from './locales/he/cgm.json';
import mood_he from './locales/he/mood.json';
import food_he from './locales/he/food.json';
import chat_he from './locales/he/chat.json';

import common_ar from './locales/ar/common.json';
import auth_ar from './locales/ar/auth.json';
import cgm_ar from './locales/ar/cgm.json';
import mood_ar from './locales/ar/mood.json';
import food_ar from './locales/ar/food.json';
import chat_ar from './locales/ar/chat.json';

import common_de from './locales/de/common.json';
import auth_de from './locales/de/auth.json';
import cgm_de from './locales/de/cgm.json';
import mood_de from './locales/de/mood.json';
import food_de from './locales/de/food.json';
import chat_de from './locales/de/chat.json';

import common_fr from './locales/fr/common.json';
import auth_fr from './locales/fr/auth.json';
import cgm_fr from './locales/fr/cgm.json';
import mood_fr from './locales/fr/mood.json';
import food_fr from './locales/fr/food.json';
import chat_fr from './locales/fr/chat.json';

import common_es from './locales/es/common.json';
import auth_es from './locales/es/auth.json';
import cgm_es from './locales/es/cgm.json';
import mood_es from './locales/es/mood.json';
import food_es from './locales/es/food.json';
import chat_es from './locales/es/chat.json';

import common_pt from './locales/pt/common.json';
import auth_pt from './locales/pt/auth.json';
import cgm_pt from './locales/pt/cgm.json';
import mood_pt from './locales/pt/mood.json';
import food_pt from './locales/pt/food.json';
import chat_pt from './locales/pt/chat.json';

import common_nl from './locales/nl/common.json';
import auth_nl from './locales/nl/auth.json';
import cgm_nl from './locales/nl/cgm.json';
import mood_nl from './locales/nl/mood.json';
import food_nl from './locales/nl/food.json';
import chat_nl from './locales/nl/chat.json';

import common_tr from './locales/tr/common.json';
import auth_tr from './locales/tr/auth.json';
import cgm_tr from './locales/tr/cgm.json';
import mood_tr from './locales/tr/mood.json';
import food_tr from './locales/tr/food.json';
import chat_tr from './locales/tr/chat.json';

import common_ru from './locales/ru/common.json';
import auth_ru from './locales/ru/auth.json';
import cgm_ru from './locales/ru/cgm.json';
import mood_ru from './locales/ru/mood.json';
import food_ru from './locales/ru/food.json';
import chat_ru from './locales/ru/chat.json';

import common_ja from './locales/ja/common.json';
import auth_ja from './locales/ja/auth.json';
import cgm_ja from './locales/ja/cgm.json';
import mood_ja from './locales/ja/mood.json';
import food_ja from './locales/ja/food.json';
import chat_ja from './locales/ja/chat.json';

import common_zh from './locales/zh/common.json';
import auth_zh from './locales/zh/auth.json';
import cgm_zh from './locales/zh/cgm.json';
import mood_zh from './locales/zh/mood.json';
import food_zh from './locales/zh/food.json';
import chat_zh from './locales/zh/chat.json';

export interface LocaleInfo {
  code: string;
  locale: string;
  label: string;
  dir: 'ltr' | 'rtl';
}

export const SUPPORTED_LOCALES: LocaleInfo[] = [
  { code: 'en', locale: 'en-US', label: 'English', dir: 'ltr' },
  { code: 'he', locale: 'he-IL', label: 'עברית', dir: 'rtl' },
  { code: 'ar', locale: 'ar-SA', label: 'العربية', dir: 'rtl' },
  { code: 'de', locale: 'de-DE', label: 'Deutsch', dir: 'ltr' },
  { code: 'fr', locale: 'fr-FR', label: 'Français', dir: 'ltr' },
  { code: 'es', locale: 'es-ES', label: 'Español', dir: 'ltr' },
  { code: 'pt', locale: 'pt-BR', label: 'Português', dir: 'ltr' },
  { code: 'nl', locale: 'nl-NL', label: 'Nederlands', dir: 'ltr' },
  { code: 'tr', locale: 'tr-TR', label: 'Türkçe', dir: 'ltr' },
  { code: 'ru', locale: 'ru-RU', label: 'Русский', dir: 'ltr' },
  { code: 'ja', locale: 'ja-JP', label: '日本語', dir: 'ltr' },
  { code: 'zh', locale: 'zh-CN', label: '中文', dir: 'ltr' },
];

export const RTL_LOCALES = ['he', 'ar'];

function getBrowserLocale(): string {
  const browserLang = navigator.language.split('-')[0];
  return SUPPORTED_LOCALES.some((l) => l.code === browserLang) ? browserLang : 'en';
}

function detectLocale(): string {
  const stored = localStorage.getItem('delight_locale');
  if (stored && SUPPORTED_LOCALES.some((l) => l.code === stored)) {
    return stored;
  }
  return getBrowserLocale();
}

function applyDirection(lng: string) {
  const localeInfo = SUPPORTED_LOCALES.find((l) => l.code === lng);
  document.documentElement.dir = localeInfo?.dir || 'ltr';
  document.documentElement.lang = localeInfo?.locale || lng;
}

i18n.use(initReactI18next).init({
  resources: {
    en: { common: common_en, auth: auth_en, cgm: cgm_en, mood: mood_en, food: food_en, chat: chat_en },
    he: { common: common_he, auth: auth_he, cgm: cgm_he, mood: mood_he, food: food_he, chat: chat_he },
    ar: { common: common_ar, auth: auth_ar, cgm: cgm_ar, mood: mood_ar, food: food_ar, chat: chat_ar },
    de: { common: common_de, auth: auth_de, cgm: cgm_de, mood: mood_de, food: food_de, chat: chat_de },
    fr: { common: common_fr, auth: auth_fr, cgm: cgm_fr, mood: mood_fr, food: food_fr, chat: chat_fr },
    es: { common: common_es, auth: auth_es, cgm: cgm_es, mood: mood_es, food: food_es, chat: chat_es },
    pt: { common: common_pt, auth: auth_pt, cgm: cgm_pt, mood: mood_pt, food: food_pt, chat: chat_pt },
    nl: { common: common_nl, auth: auth_nl, cgm: cgm_nl, mood: mood_nl, food: food_nl, chat: chat_nl },
    tr: { common: common_tr, auth: auth_tr, cgm: cgm_tr, mood: mood_tr, food: food_tr, chat: chat_tr },
    ru: { common: common_ru, auth: auth_ru, cgm: cgm_ru, mood: mood_ru, food: food_ru, chat: chat_ru },
    ja: { common: common_ja, auth: auth_ja, cgm: cgm_ja, mood: mood_ja, food: food_ja, chat: chat_ja },
    zh: { common: common_zh, auth: auth_zh, cgm: cgm_zh, mood: mood_zh, food: food_zh, chat: chat_zh },
  },
  lng: detectLocale(),
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false,
  },
  react: {
    useSuspense: false,
  },
});

i18n.on('languageChanged', (lng) => {
  localStorage.setItem('delight_locale', lng);
  applyDirection(lng);
});

// Set initial direction on load
applyDirection(i18n.language);

export default i18n;
