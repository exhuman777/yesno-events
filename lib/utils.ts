import { type ClassValue, clsx } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

export function formatCurrency(amount: number): string {
  return `$${amount.toLocaleString()}`;
}

export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

export function shuffleArray<T>(array: T[]): T[] {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

export function selectRandom<T>(array: T[], count: number): T[] {
  return shuffleArray(array).slice(0, count);
}

export function formatAnomalyScore(score: number): string {
  if (score <= 0) return '';

  // Cap display at 300% for readability
  const percentage = Math.min(score * 100, 300);

  // Show as integer percentage
  return `+${Math.round(percentage)}%`;
}
