import { useTranslation } from 'react-i18next';

interface CGMSummaryCardProps {
  avgGlucose: number;
  minGlucose: number;
  maxGlucose: number;
  timeInRangePct: number;
  spikeCount: number;
  hypoCount: number;
}

const getGlucoseColor = (value: number) => {
  if (value < 70 || value > 180) return 'text-red-600';
  if (value > 140) return 'text-yellow-600';
  return 'text-green-600';
};

const getTIRColor = (pct: number) => {
  if (pct < 50) return 'text-red-600';
  if (pct < 70) return 'text-yellow-600';
  return 'text-green-600';
};

export default function CGMSummaryCard({
  avgGlucose,
  minGlucose,
  maxGlucose,
  timeInRangePct,
  spikeCount,
  hypoCount,
}: CGMSummaryCardProps) {
  const { t, i18n } = useTranslation('cgm');

  // Locale-aware number formatting via Intl.NumberFormat
  const fmtGlucose = (n: number) =>
    new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 0 }).format(n);
  const fmtPct = (n: number) =>
    new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 1 }).format(n);
  const fmtCount = (n: number) =>
    new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 0 }).format(n);

  const stats = [
    { label: t('stats.avg'), value: fmtGlucose(avgGlucose), unit: 'mg/dL', color: getGlucoseColor(avgGlucose), tooltip: t('tooltips.avg') },
    { label: t('stats.min'), value: fmtGlucose(minGlucose), unit: 'mg/dL', color: getGlucoseColor(minGlucose), tooltip: t('tooltips.min') },
    { label: t('stats.max'), value: fmtGlucose(maxGlucose), unit: 'mg/dL', color: getGlucoseColor(maxGlucose), tooltip: t('tooltips.max') },
    { label: t('stats.tir'), value: fmtPct(timeInRangePct), unit: '%', color: getTIRColor(timeInRangePct), tooltip: t('tooltips.tir') },
    { label: t('stats.spikes'), value: fmtCount(spikeCount), unit: '', color: spikeCount > 0 ? 'text-red-600' : 'text-green-600', tooltip: t('tooltips.spikes') },
    { label: t('stats.hypos'), value: fmtCount(hypoCount), unit: '', color: hypoCount > 0 ? 'text-red-600' : 'text-green-600', tooltip: t('tooltips.hypos') },
  ];

  return (
    <div className="grid grid-cols-3 md:grid-cols-6 gap-3" role="group" aria-label="CGM Summary">
      {stats.map((stat) => (
        <div
          key={stat.label}
          className="bg-slate-50 rounded-lg p-3 text-center hover:bg-slate-100 transition-colors cursor-default"
          title={stat.tooltip}
        >
          <div className={`text-xl font-bold ${stat.color}`}>
            {stat.value}
            <span className="text-xs ms-1">{stat.unit}</span>
          </div>
          <div className="text-xs text-slate-500 mt-1">{stat.label}</div>
        </div>
      ))}
    </div>
  );
}
