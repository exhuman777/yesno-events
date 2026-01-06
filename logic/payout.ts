import type { Participant, PayoutResult } from '@/types/market';

/**
 * Calculate payouts based on hit distribution
 */
export function calculatePayouts(
  participants: Participant[],
  prizePool: number
): PayoutResult[] {
  const totalHits = participants.reduce((sum, p) => sum + p.totalHits, 0);

  // No hits - refund everyone
  if (totalHits === 0) {
    return participants.map((p, index) => ({
      participantId: p.id,
      payout: p.buyIn,
      rank: index + 1,
      hitShare: 0,
    }));
  }

  // Sort by total hits (descending)
  const sorted = [...participants].sort((a, b) => b.totalHits - a.totalHits);

  // Calculate proportional payouts
  const results: PayoutResult[] = sorted.map((p, index) => {
    const hitShare = p.totalHits / totalHits;
    const payout = prizePool * hitShare;

    return {
      participantId: p.id,
      payout,
      rank: index + 1,
      hitShare,
    };
  });

  return results;
}

/**
 * Get leaderboard sorted by hits
 */
export function getLeaderboard(participants: Participant[]): Participant[] {
  return [...participants].sort((a, b) => b.totalHits - a.totalHits);
}

/**
 * Calculate potential payout for a participant
 */
export function calculatePotentialPayout(
  participant: Participant,
  allParticipants: Participant[],
  prizePool: number
): number {
  const totalHits = allParticipants.reduce((sum, p) => sum + p.totalHits, 0);

  if (totalHits === 0) {
    return participant.buyIn;
  }

  const hitShare = participant.totalHits / totalHits;
  return prizePool * hitShare;
}
