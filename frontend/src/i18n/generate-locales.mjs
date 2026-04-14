// Script to generate all translation files for Delight Health i18n
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const locales = ['en', 'he', 'ar', 'de', 'fr', 'es', 'pt', 'nl', 'tr', 'ru', 'ja', 'zh'];

// English base translations (already created, using as template)
const enTranslations = {
  common: {
    save: 'Save',
    cancel: 'Cancel',
    loading: 'Loading...',
    error: 'Error',
    back: 'Back',
    next: 'Next',
    yes: 'Yes',
    no: 'No',
    skip: 'Skip',
    continue: 'Continue',
    submit: 'Submit',
    delete: 'Delete',
    edit: 'Edit',
    add: 'Add',
    remove: 'Remove',
    search: 'Search',
    filter: 'Filter',
    close: 'Close',
    open: 'Open',
    status: { active: 'Active', suspended: 'Suspended', pending: 'Pending' },
    timeAgo: '{{n}} {{unit}} ago',
    todayAt: 'Today at {{time}}',
    units: { hours: 'hours', minutes: 'minutes', days: 'days' }
  },
  auth: {
    email: 'Email',
    password: 'Password',
    firstName: 'First Name',
    lastName: 'Last Name',
    signIn: 'Sign In',
    register: 'Register',
    noAccount: "Don't have an account?",
    hasAccount: 'Already have an account?',
    signingIn: 'Signing in...',
    showPassword: 'Show password',
    hidePassword: 'Hide password',
    welcomeBack: 'Welcome back to Delight Health',
    logout: 'Logout',
    emailPlaceholder: 'name@email.com',
    passwordPlaceholder: 'Enter your password',
    invalidCredentials: 'Invalid credentials',
    loggedIn: 'Logged in successfully'
  },
  cgm: {
    unit: { mgdl: 'mg/dL', mmoll: 'mmol/L' },
    chart: { yAxisLabel: 'Glucose ({{unit}})', ariaLabel: 'Glucose readings chart, last {{period}}. Target range shaded green.' },
    referenceLines: { hypo: 'Hypoglycaemia threshold', target: 'Target range upper limit', hyper: 'Hyperglycaemia threshold' },
    period: { '24h': '24h', '7d': '7d', '30d': '30d' },
    empty: 'No glucose data yet',
    addReading: 'Add your first reading',
    stats: { avg: 'Avg Glucose', min: 'Min', max: 'Max', tir: 'Time in Range', spikes: 'Spikes', hypos: 'Hypos' },
    tooltips: {
      avg: 'Average glucose over the selected period',
      min: 'Lowest reading in period',
      max: 'Highest reading in period',
      tir: 'Percentage of time glucose was within your target range',
      spikes: 'Number of readings above 180 mg/dL (hyperglycaemic events)',
      hypos: 'Number of readings below 70 mg/dL (hypoglycaemic events). Contact your doctor if frequent.'
    },
    manualEntry: {
      placeholder: 'Enter glucose ({{unit}})',
      add: 'Add',
      trend: 'Trend',
      deviceId: 'Device ID (optional)',
      deviceIdPlaceholder: 'e.g. Libre 3',
      timestamp: 'Timestamp',
      trends: { risingRapidly: 'Rising Rapidly', rising: 'Rising', stable: 'Stable', falling: 'Falling', fallingRapidly: 'Falling Rapidly' }
    },
    range: { hypo: 'Hypo', normal: 'Normal', high: 'High' },
    saved: 'Glucose reading saved successfully',
    saveFailed: 'Failed to save glucose reading'
  },
  mood: {
    title: 'How are you feeling?',
    emotions: { happy: 'Happy', calm: 'Calm', energised: 'Energised', tired: 'Tired', anxious: 'Anxious', sad: 'Sad', irritable: 'Irritable', unwell: 'Unwell', other: 'Other' },
    stress: { label: 'Stress Level', low: 'Low', moderate: 'Moderate', high: 'High', ariaValue: '{{n}} out of {{max}}' },
    notesPlaceholder: 'Optional notes about how you feel...',
    logMood: 'Log Mood',
    moodLogged: 'Mood logged successfully',
    logFailed: 'Failed to log mood',
    selectEmotion: 'Select an emotion',
    aiInsight: 'Delight insight:',
    describeFeelings: 'Describe how you feel...'
  },
  food: {
    title: 'Log a Meal',
    meal: { breakfast: 'Breakfast', lunch: 'Lunch', dinner: 'Dinner', snack: 'Snack' },
    manualEntry: {
      foodInput: 'Add food item (Enter to add)',
      calories: 'Estimated Calories',
      kcal: 'kcal',
      glycemicLoad: 'Glycaemic Load',
      gl: { low: 'Low', medium: 'Medium', high: 'High' }
    },
    photo: {
      uploadPrompt: 'Drag photo here or tap to upload',
      analysing: 'Analysing your meal...',
      timeout: 'Photo recognition timed out. Please log manually.',
      lowConfidence: 'Some items may be inaccurate. Please review before saving.',
      unavailable: 'Photo recognition unavailable. Please log manually.'
    },
    confidence: 'Confidence: {{pct}}%',
    saveToLog: 'Save to Food Log',
    logMeal: 'Log Meal',
    mealLogged: 'Meal logged successfully',
    logFailed: 'Failed to log meal',
    addFood: 'Add at least one food item',
    history: { title: 'Recent Meals', empty: 'No meals logged yet' }
  },
  chat: {
    title: 'AI Health Assistant',
    disclaimer: {
      heading: 'Medical disclaimer',
      body: 'Delight is an AI health companion, not a doctor. It does not provide medical diagnoses or replace professional advice. Always consult your doctor for medical decisions.'
    },
    inputPlaceholder: 'Type your message...',
    send: 'Send message',
    emptyState: 'No messages yet. Ask me anything about your health!',
    aiTyping: 'Delight is thinking...',
    aiError: 'AI service is temporarily unavailable. Please try again later.'
  }
};

// For production, each locale would have professional translations
// For now, we'll use English as fallback and mark them for translation review
const localeNames = {
  en: 'English',
  he: 'עברית',
  ar: 'العربية',
  de: 'Deutsch',
  fr: 'Français',
  es: 'Español',
  pt: 'Português',
  nl: 'Nederlands',
  tr: 'Türkçe',
  ru: 'Русский',
  ja: '日本語',
  zh: '中文'
};

console.log('Generating translation files for all 12 locales...');
console.log('Note: Non-English locales use English as placeholder - REPLACE WITH PROFESSIONAL TRANSLATIONS');

locales.forEach(locale => {
  const localeDir = join(__dirname, 'locales', locale);
  
  if (!existsSync(localeDir)) {
    mkdirSync(localeDir, { recursive: true });
  }
  
  Object.keys(enTranslations).forEach(namespace => {
    const filePath = join(localeDir, `${namespace}.json`);
    const content = JSON.stringify(enTranslations[namespace], null, 2);
    writeFileSync(filePath, content, 'utf8');
    console.log(`✓ Created ${locale}/${namespace}.json`);
  });
});

console.log('\n✅ All translation files generated successfully!');
console.log('\nIMPORTANT: Files for he, ar, de, fr, es, pt, nl, tr, ru, ja, zh contain English text.');
console.log('Replace with professional translations before production release.');
