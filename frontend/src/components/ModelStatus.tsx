import { useState, useEffect, useCallback } from 'react';
import { Activity, CheckCircle, XCircle, Wifi, WifiOff } from 'lucide-react';
import { api, type ModelStatusResponse } from '../api';

export default function ModelStatus() {
  const [health, setHealth] = useState<boolean | null>(null);
  const [model, setModel] = useState<ModelStatusResponse | null>(null);

  const refresh = useCallback(async () => {
    try {
      await api.health();
      setHealth(true);
    } catch {
      setHealth(false);
    }
    try {
      const m = await api.modelStatus();
      setModel(m);
    } catch {
      setModel(null);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  }, [refresh]);

  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-[var(--color-border)] bg-[var(--color-bg-secondary)]">
      <div className="flex items-center gap-3">
        <Activity className="w-6 h-6 text-[var(--color-accent)]" />
        <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
          MetroPT-3 Predictive Maintenance
        </h1>
      </div>

      <div className="flex items-center gap-4 text-xs">
        {/* Backend health */}
        <div className="flex items-center gap-1.5">
          {health === null ? (
            <div className="w-2 h-2 rounded-full bg-gray-500 animate-pulse" />
          ) : health ? (
            <Wifi className="w-3.5 h-3.5 text-[var(--color-success)]" />
          ) : (
            <WifiOff className="w-3.5 h-3.5 text-[var(--color-danger)]" />
          )}
          <span className="text-[var(--color-text-muted)]">
            API {health ? 'Connected' : health === false ? 'Offline' : '...'}
          </span>
        </div>

        {/* Model status */}
        <div className="flex items-center gap-1.5">
          {model?.exists ? (
            <CheckCircle className="w-3.5 h-3.5 text-[var(--color-success)]" />
          ) : (
            <XCircle className="w-3.5 h-3.5 text-[var(--color-text-muted)]" />
          )}
          <span className="text-[var(--color-text-muted)]">
            {model?.exists
              ? `Model trained${model.last_trained ? ` (${model.last_trained})` : ''}`
              : 'No model'}
          </span>
        </div>

        {model?.baseline_error !== null && model?.baseline_error !== undefined && (
          <span className="text-[var(--color-text-muted)]">
            Baseline: {model.baseline_error.toFixed(6)}
          </span>
        )}
      </div>
    </header>
  );
}
