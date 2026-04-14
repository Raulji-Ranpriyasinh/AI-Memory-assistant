import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  fullPage?: boolean;
}

export default function LoadingSpinner({ size = 'md', fullPage = false }: LoadingSpinnerProps) {
  const sizeMap = { sm: 'w-4 h-4', md: 'w-8 h-8', lg: 'w-12 h-12' };

  const spinner = (
    <Loader2 className={`${sizeMap[size]} animate-spin text-blue-600`} />
  );

  if (fullPage) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        {spinner}
      </div>
    );
  }

  return spinner;
}
