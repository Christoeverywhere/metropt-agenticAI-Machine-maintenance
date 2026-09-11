/**
 * API client for the MetroPT-3 backend service.
 * Uses environment-based base URL, never hardcoded localhost.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  message: string;
}

export interface DatasetUploadResponse {
  status: string;
  row_count: number;
  date_start: string;
  date_end: string;
  columns: string[];
}

export interface SensorMissing {
  sensor: string;
  missing_count: number;
  missing_pct: number;
}

export interface DatasetSummaryResponse {
  row_count: number;
  date_start: string;
  date_end: string;
  healthy_train_rows: number;
  stream_rows: number;
  missing_values: SensorMissing[];
}

export interface TrainStartResponse {
  job_id: string;
  message: string;
}

export interface TrainStatusResponse {
  job_id: string;
  status: string;
  stage: number | null;
  epoch: number | null;
  total_epochs: number | null;
  chunk: number | null;
  total_chunks: number | null;
  latest_loss: number | null;
  baseline_error: number | null;
  logs: string[];
  error: string | null;
}

export interface EvalStartResponse {
  job_id: string;
  message: string;
}

export interface EvalStatusResponse {
  job_id: string;
  status: string;
  logs: string[];
  error: string | null;
}

export interface SensorContribution {
  sensor: string;
  error: number;
}

export interface FailureExplanation {
  failure_id: number;
  failure_start: string;
  top_sensors: SensorContribution[];
}

export interface LeadTimeResult {
  failure_id: number;
  failure_start: string;
  failure_end: string;
  failure_type: string;
  severity: string;
  detected: boolean;
  first_flag: string | null;
  lead_time_seconds: number | null;
  lead_time_str: string;
}

export interface EvalResultsResponse {
  threshold: number;
  mu: number;
  sigma: number;
  baseline_error: number;
  lead_times: LeadTimeResult[];
  explanations: FailureExplanation[];
  n_scored_rows: number;
  n_flagged: number;
}

export interface ScorePoint {
  timestamp: string;
  seq_error: number;
  rolling_score?: number;
  flagged?: boolean;
  [key: string]: string | number | boolean | undefined;
}

export interface ScoresResponse {
  points: ScorePoint[];
  total_rows: number;
  returned_rows: number;
  start: string;
  end: string;
}

export interface SensorDataResponse {
  sensor: string;
  scaled: boolean;
  points: { timestamp: string; value: number }[];
  total_rows: number;
  returned_rows: number;
}

export interface FailureWindow {
  id: number;
  start: string;
  end: string;
  type: string;
  severity: string;
}

export interface FailuresResponse {
  failures: FailureWindow[];
}

export interface ModelStatusResponse {
  exists: boolean;
  last_trained: string | null;
  baseline_error: number | null;
  scaler_exists: boolean;
  scored_stream_exists: boolean;
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `API error ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const api = {
  health: () => apiFetch<HealthResponse>('/api/health'),

  // Dataset
  uploadDataset: async (file: File): Promise<DatasetUploadResponse> => {
    const form = new FormData();
    form.append('file', file);
    return apiFetch<DatasetUploadResponse>('/api/dataset/upload', {
      method: 'POST',
      body: form,
    });
  },
  datasetSummary: () => apiFetch<DatasetSummaryResponse>('/api/dataset/summary'),

  // Training
  startTraining: () =>
    apiFetch<TrainStartResponse>('/api/train/start', { method: 'POST' }),
  trainingStatus: (jobId: string) =>
    apiFetch<TrainStatusResponse>(`/api/train/status/${jobId}`),

  // Evaluation
  startEvaluation: () =>
    apiFetch<EvalStartResponse>('/api/evaluate/run', { method: 'POST' }),
  evaluationStatus: (jobId: string) =>
    apiFetch<EvalStatusResponse>(`/api/evaluate/status/${jobId}`),
  evaluationResults: () => apiFetch<EvalResultsResponse>('/api/evaluate/results'),

  // Scores & data
  getScores: (params: {
    start?: string;
    end?: string;
    stride?: number;
    max_points?: number;
  } = {}) => {
    const qs = new URLSearchParams();
    if (params.start) qs.set('start', params.start);
    if (params.end) qs.set('end', params.end);
    if (params.stride) qs.set('stride', String(params.stride));
    if (params.max_points) qs.set('max_points', String(params.max_points));
    return apiFetch<ScoresResponse>(`/api/scores?${qs.toString()}`);
  },

  getRawSensor: (params: {
    sensor: string;
    start?: string;
    end?: string;
    scaled?: boolean;
    max_points?: number;
  }) => {
    const qs = new URLSearchParams();
    qs.set('sensor', params.sensor);
    if (params.start) qs.set('start', params.start);
    if (params.end) qs.set('end', params.end);
    if (params.scaled !== undefined) qs.set('scaled', String(params.scaled));
    if (params.max_points) qs.set('max_points', String(params.max_points));
    return apiFetch<SensorDataResponse>(`/api/sensors/raw?${qs.toString()}`);
  },

  getFailures: () => apiFetch<FailuresResponse>('/api/failures'),

  // Model
  modelStatus: () => apiFetch<ModelStatusResponse>('/api/model/status'),
};
