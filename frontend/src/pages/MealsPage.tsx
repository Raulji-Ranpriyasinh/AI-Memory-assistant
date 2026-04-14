import { useState, useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import api from '../api/client';
import toast from 'react-hot-toast';
import { Upload, Trash2, Camera, ImagePlus } from 'lucide-react';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const MEAL_TYPES = ['breakfast', 'lunch', 'dinner', 'snack'] as const;

export default function MealsPage() {
  const { t: tf } = useTranslation('food');
  const [selectedMeal, setSelectedMeal] = useState('lunch');
  const [foodItems, setFoodItems] = useState<string[]>([]);
  const [foodInput, setFoodInput] = useState('');
  const [foodHistory, setFoodHistory] = useState<any[]>([]);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadFoodHistory();
  }, []);

  const loadFoodHistory = async () => {
    setHistoryLoading(true);
    try {
      const res = await api.get('/food/history');
      setFoodHistory(res.data.data || []);
    } catch {
      // Ignore errors
    } finally {
      setHistoryLoading(false);
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

  const handleImageSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      toast.error('Image must be less than 10MB');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setImageBase64(result);
      setImagePreview(result);
    };
    reader.readAsDataURL(file);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleImageSelect(file);
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleImageSelect(file);
  }, []);

  const handleRemoveImage = () => {
    setImageBase64(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleLogMeal = async () => {
    if (foodItems.length === 0 && !imageBase64) {
      toast.error(tf('addFood'));
      return;
    }

    setLoading(true);
    try {
      const payload: any = {
        mealType: selectedMeal,
        items: foodItems,
        timestamp: new Date().toISOString(),
      };

      if (imageBase64) {
        // Use base64 image data
        payload.imageBase64 = imageBase64;
      }

      await api.post('/food/log', payload);
      toast.success(tf('mealLogged'));
      setFoodItems([]);
      setImageBase64(null);
      setImagePreview(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      loadFoodHistory();
    } catch {
      toast.error(tf('logFailed'));
    } finally {
      setLoading(false);
    }
  };

  // Group meals by date
  const groupedMeals = foodHistory.reduce((acc: Record<string, any[]>, meal) => {
    const date = new Date(meal.createdAt || meal.timestamp).toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
    if (!acc[date]) acc[date] = [];
    acc[date].push(meal);
    return acc;
  }, {});

  return (
    <div className="p-4 md:p-6 space-y-6 max-w-4xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl md:text-3xl font-bold text-slate-800">
          <span className="bg-gradient-to-r from-amber-500 to-orange-500 bg-clip-text text-transparent">
            Meal Logger
          </span>
        </h1>
        <p className="text-slate-500 mt-1">Track your meals and nutrition</p>
      </div>

      {/* Log a Meal Section */}
      <div className="card p-5">
        <h2 className="text-lg font-bold text-slate-800 mb-4">{tf('title')}</h2>

        {/* Meal Type Selector */}
        <div className="flex flex-wrap gap-2 mb-4">
          {MEAL_TYPES.map((meal) => (
            <button
              key={meal}
              onClick={() => setSelectedMeal(meal)}
              className={`px-4 py-2.5 rounded-xl capitalize transition-all duration-200 font-medium text-sm min-h-[44px] ${
                selectedMeal === meal
                  ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-md'
                  : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
              }`}
            >
              {tf(`meal.${meal}`)}
            </button>
          ))}
        </div>

        {/* Image Upload Zone */}
        <div className="mb-4">
          {!imagePreview ? (
            <div
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
                dragActive
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileInputChange}
                className="hidden"
              />
              <ImagePlus className="w-12 h-12 text-slate-400 mx-auto mb-3" />
              <p className="text-slate-600 font-medium mb-1">
                Drag & drop a meal photo here, or <span className="text-blue-600">browse</span>
              </p>
              <p className="text-slate-400 text-xs">PNG, JPG, WEBP up to 10MB</p>
            </div>
          ) : (
            <div className="relative rounded-xl overflow-hidden border-2 border-slate-200">
              <img
                src={imagePreview}
                alt="Meal preview"
                className="w-full h-48 object-cover"
              />
              <button
                onClick={handleRemoveImage}
                className="absolute top-2 right-2 w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600 transition-colors shadow-lg"
                aria-label="Remove image"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>

        {/* Food Items Input */}
        <div className="mb-4">
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Food items
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={foodInput}
              onChange={(e) => setFoodInput(e.target.value)}
              placeholder={tf('manualEntry.foodInput')}
              className="flex-1 px-4 py-3 border-2 border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200"
              onKeyDown={(e) => e.key === 'Enter' && handleAddFoodItem()}
              dir="auto"
            />
            <button
              onClick={handleAddFoodItem}
              className="bg-slate-100 text-slate-700 px-4 py-3 rounded-xl hover:bg-slate-200 transition-colors min-h-[44px] font-medium"
            >
              Add
            </button>
          </div>
        </div>

        {/* Food Items Tags */}
        {foodItems.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {foodItems.map((item, i) => (
              <span
                key={i}
                className="bg-blue-100 text-blue-800 px-3 py-1.5 rounded-full text-sm flex items-center gap-2 font-medium"
              >
                {item}
                <button
                  onClick={() => handleRemoveFoodItem(i)}
                  className="hover:text-blue-600 transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                  aria-label={`Remove ${item}`}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </span>
            ))}
          </div>
        )}

        {/* Submit Button */}
        <button
          onClick={handleLogMeal}
          disabled={loading}
          className="w-full bg-gradient-to-r from-amber-500 to-orange-500 text-white py-3 rounded-xl hover:from-amber-600 hover:to-orange-600 disabled:opacity-50 disabled:cursor-not-allowed font-semibold shadow-md hover:shadow-lg transition-all duration-200 min-h-[44px]"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <LoadingSpinner size="sm" />
              Logging meal...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <Upload className="w-5 h-5" />
              {tf('logMeal')}
            </span>
          )}
        </button>
      </div>

      {/* Meal History */}
      <div className="card p-5">
        <h2 className="text-lg font-bold text-slate-800 mb-4">{tf('history.title')}</h2>

        {historyLoading ? (
          <div className="flex justify-center py-8">
            <LoadingSpinner />
          </div>
        ) : Object.keys(groupedMeals).length === 0 ? (
          <div className="text-center py-8">
            <Camera className="w-12 h-12 text-slate-400 mx-auto mb-3" />
            <p className="text-slate-500 font-medium">No meals logged yet</p>
            <p className="text-slate-400 text-sm">Start by logging your first meal above</p>
          </div>
        ) : (
          <div className="space-y-6">
            {Object.entries(groupedMeals).map(([date, meals]) => (
              <div key={date}>
                <h3 className="text-sm font-semibold text-slate-500 mb-3 pb-2 border-b border-slate-200">
                  {date}
                </h3>
                <div className="space-y-3">
                  {meals.map((meal: any, i: number) => (
                    <div
                      key={i}
                      className="bg-gradient-to-r from-slate-50 to-white rounded-xl border border-slate-200 overflow-hidden"
                    >
                      {/* Meal Image Thumbnail */}
                      {(meal.imageUrl || meal.imageBase64) && (
                        <div className="h-32 overflow-hidden">
                          <img
                            src={meal.imageBase64 || meal.imageUrl}
                            alt={meal.mealType}
                            className="w-full h-full object-cover"
                          />
                        </div>
                      )}
                      <div className="p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="px-3 py-1 bg-gradient-to-r from-amber-100 to-orange-100 text-amber-800 rounded-full text-xs font-semibold capitalize">
                            {tf(`meal.${meal.mealType}`)}
                          </span>
                          <span className="text-xs text-slate-400">
                            {new Date(meal.createdAt || meal.timestamp).toLocaleTimeString([], {
                              hour: '2-digit',
                              minute: '2-digit',
                            })}
                          </span>
                        </div>
                        <p className="text-sm text-slate-700">
                          {(meal.items || []).join(', ')}
                        </p>
                        {meal.glycemicLoad && (
                          <span className="inline-block mt-2 px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">
                            Glycemic load: {meal.glycemicLoad}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
