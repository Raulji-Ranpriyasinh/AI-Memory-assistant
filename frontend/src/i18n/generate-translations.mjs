import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';

const __dirname = dirname(new URL(import.meta.url).pathname);

const translations: Record<string, Record<string, Record<string, any>>> = {
  he: {
    common: {
      save: 'שמור',
      cancel: 'ביטול',
      loading: 'טוען...',
      error: 'שגיאה',
      back: 'חזור',
      next: 'הבא',
      yes: 'כן',
      no: 'לא',
      skip: 'דלג',
      continue: 'המשך',
      submit: 'שלח',
      delete: 'מחק',
      edit: 'ערוך',
      add: 'הוסף',
      remove: 'הסר',
      search: 'חפש',
      filter: 'סנן',
      close: 'סגור',
      open: 'פתח',
      status: { active: 'פעיל', suspended: 'מושעה', pending: 'ממתין' },
      timeAgo: 'לפני {{n}} {{unit}}',
      todayAt: 'היום ב-{{time}}',
      units: { hours: 'שעות', minutes: 'דקות', days: 'ימים' }
    },
    auth: {
      email: 'דוא"ל',
      password: 'סיסמה',
      firstName: 'שם פרטי',
      lastName: 'שם משפחה',
      signIn: 'התחבר',
      register: 'הרשם',
      noAccount: 'אין לך חשבון?',
      hasAccount: 'כבר יש לך חשבון?',
      signingIn: 'מתחבר...',
      showPassword: 'הצג סיסמה',
      hidePassword: 'הסתר סיסמה',
      welcomeBack: 'ברוך הבא ל-Delight Health',
      logout: 'התנתק',
      emailPlaceholder: 'name@email.com',
      passwordPlaceholder: 'הכנס סיסמה',
      invalidCredentials: 'פרטי התחברות שגויים',
      loggedIn: 'התחברת בהצלחה'
    },
    cgm: {
      unit: { mgdl: 'מ"ג/ד"ל', mmoll: 'ממול/ל' },
      period: { '24h': '24 שעות', '7d': '7 ימים', '30d': '30 ימים' },
      empty: 'אין נתוני גלוקוז עדיין',
      addReading: 'הוסף את המדידה הראשונה שלך',
      stats: { avg: 'ממוצע גלוקוז', min: 'מינימום', max: 'מקסימום', tir: 'זמן בטווח', spikes: 'עליות חדות', hypos: 'ירידות חדות' },
      saved: 'מדידת גלוקוז נשמרה בהצלחה',
      saveFailed: 'שמירת מדידת גלוקוז נכשלה'
    },
    mood: {
      title: 'איך אתה מרגיש?',
      emotions: { happy: 'שמח', calm: 'רגוע', energised: 'מלא אנרגיה', tired: 'עייף', anxious: 'חרד', sad: 'עצוב', irritable: 'רגזן', unwell: 'לא מרגיש טוב', other: 'אחר' },
      stress: { label: 'רמת לחץ', low: 'נמוך', moderate: 'בינוני', high: 'גבוה', ariaValue: '{{n}} מתוך {{max}}' },
      notesPlaceholder: 'הערות אופציונליות...',
      logMood: 'תעד מצב רוח',
      moodLogged: 'מצב הרוח נרשם בהצלחה',
      logFailed: 'רישום מצב רוח נכשל',
      selectEmotion: 'בחר רגש',
      aiInsight: 'תובנת Delight:',
      describeFeelings: 'תאר איך אתה מרגיש...'
    },
    food: {
      title: 'תעד ארוחה',
      meal: { breakfast: 'ארוחת בוקר', lunch: 'ארוחת צהריים', dinner: 'ארוחת ערב', snack: 'חטיף' },
      manualEntry: { foodInput: 'הוסף פריט מזון (Enter להוספה)', calories: 'קלוריות משוערות', kcal: 'קק"ל' },
      logMeal: 'תעד ארוחה',
      mealLogged: 'הארוחה נרשמה בהצלחה',
      logFailed: 'רישום הארוחה נכשל',
      addFood: 'הוסף לפחות פריט מזון אחד',
      history: { title: 'ארוחות אחרונות', empty: 'עדיין לא נרשמו ארוחות' }
    },
    chat: {
      title: 'עוזר בריאות AI',
      disclaimer: { heading: 'הבהרה רפואית', body: 'Delight הוא עוזר בריאות מבוסס AI, לא רופא. הוא אינו מספק אבחונים רפואיים או תחליף לייעוץ מקצועי. תמיד התייעץ עם הרופא שלך להחלטות רפואיות.' },
      inputPlaceholder: 'הקלד הודעה...',
      send: 'שלח הודעה',
      emptyState: 'עדיין אין הודעות. שאל אותי הכל על הבריאות שלך!',
      aiTyping: 'Delight חושב...',
      aiError: 'שירות ה-AI אינו זמין כרגע. אנסה שוב מאוחר יותר.'
    }
  },
  // For other locales, we'll use English as base with locale-specific adjustments
  // A production app would have professional translations for each
};

// Generate files for locales that don't have custom translations
const locales = ['ar', 'de', 'fr', 'es', 'pt', 'nl', 'tr', 'ru', 'ja', 'zh'];

// Base English template for reference
const enTemplate = {
  common: require('./locales/en/common.json'),
  auth: require('./locales/en/auth.json'),
  cgm: require('./locales/en/cgm.json'),
  mood: require('./locales/en/mood.json'),
  food: require('./locales/en/food.json'),
  chat: require('./locales/en/chat.json'),
};

console.log('Translation files generation started...');
console.log('Note: For production, all non-English translations should be professionally translated.');
console.log('Generating structure for all 12 locales...');
