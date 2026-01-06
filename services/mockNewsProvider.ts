import { BaseNewsProvider } from './newsProvider';
import type { NewsEvent, MockNewsConfig } from '@/types/news';
import { generateId } from '@/lib/utils';

/**
 * Mock news provider that generates synthetic headlines
 */
export class MockNewsProvider extends BaseNewsProvider {
  private config: MockNewsConfig;
  private intervalId: NodeJS.Timeout | null = null;

  // Professional headline templates organized by category
  private headlines = {
    finance: [
      'Wall Street reacts to {WORD} developments amid market volatility',
      '{WORD} trading volume hits record high as investors seek opportunities',
      'Federal Reserve monitors {WORD} indicators for policy decisions',
      'Global markets respond to unexpected {WORD} announcement',
      'Analysts upgrade {WORD} outlook following strong earnings data',
      '{WORD} futures surge on optimistic economic forecast',
      'Major institutional investors increase {WORD} exposure',
      'Treasury yields shift as {WORD} concerns dominate trading floor',
    ],
    politics: [
      'Lawmakers debate comprehensive {WORD} reform legislation',
      '{WORD} takes center stage in heated congressional hearing',
      'Bipartisan coalition forms to address {WORD} challenges',
      'White House unveils ambitious {WORD} strategy for next term',
      'Supreme Court to review landmark {WORD} case next session',
      'International summit focuses on cooperative {WORD} solutions',
      'Grassroots movement gains momentum on {WORD} platform',
      'Poll numbers shift as candidates clash over {WORD} policy',
    ],
    tech: [
      'Silicon Valley startup disrupts {WORD} industry with innovation',
      '{WORD} breakthrough announced at major technology conference',
      'Cybersecurity experts warn of emerging {WORD} vulnerabilities',
      'Tech giants collaborate on open-source {WORD} initiative',
      'Regulatory scrutiny intensifies around {WORD} practices',
      'Venture capital pours billions into {WORD} development',
      'Research team achieves milestone in {WORD} capabilities',
      'Industry standards updated to address {WORD} advancements',
    ],
    climate: [
      'Scientists report accelerating {WORD} trends in latest assessment',
      'International coalition commits to {WORD} reduction targets',
      '{WORD} reaches critical threshold, experts urge immediate action',
      'Renewable energy sector responds to {WORD} challenges',
      'New satellite data reveals extent of {WORD} impact',
      'Coastal communities prepare for {WORD} consequences',
      'Innovative {WORD} mitigation technology shows promising results',
      'Global conference addresses urgent {WORD} timeline',
    ],
    health: [
      'Clinical trials demonstrate efficacy of new {WORD} treatment',
      'Public health officials monitor {WORD} transmission patterns',
      'FDA approves accelerated review process for {WORD} therapies',
      'Research institutions collaborate on {WORD} prevention strategies',
      'Healthcare systems implement updated {WORD} protocols',
      'Breakthrough study links {WORD} to unexpected health outcomes',
      'Medical community debates optimal {WORD} intervention timing',
      'Insurance coverage expands for {WORD} related procedures',
    ],
    general: [
      'Experts analyze long-term implications of {WORD} developments',
      'Breaking: {WORD} announcement sends shockwaves through industry',
      'Investigative report uncovers previously unknown {WORD} connections',
      'International community responds to escalating {WORD} situation',
      'Leading authorities issue updated guidance on {WORD} matters',
      'Unprecedented {WORD} event prompts emergency response measures',
    ],
  };

  private contentTemplates = [
    'Industry analysts have been closely monitoring {WORD} indicators throughout the quarter. Recent developments suggest a significant shift in market dynamics that could reshape strategic planning across multiple sectors.',
    'According to sources familiar with the matter, {WORD} has become a top priority for decision makers. The rapid pace of change has caught many stakeholders off guard, prompting urgent reassessment of existing frameworks.',
    'Data compiled from multiple research institutions indicates that {WORD} trends are accelerating faster than previously projected. Experts emphasize the need for coordinated response and adaptive strategies moving forward.',
    'Stakeholders gathered for an emergency session to discuss {WORD} implications. The consensus emerging from these deliberations points to fundamental changes in how organizations approach long-term planning.',
    'New evidence has emerged regarding {WORD} that challenges conventional assumptions. Independent verification is underway, but preliminary findings suggest the need for immediate attention from relevant authorities.',
    'Following extensive analysis, specialists have identified {WORD} as a critical factor influencing current outcomes. Risk assessment models are being recalibrated to account for these newly recognized variables.',
    'The latest quarterly review highlights {WORD} as an area of particular concern. Compliance teams are working around the clock to ensure alignment with evolving regulatory expectations and industry best practices.',
    'Observers note that {WORD} has reached an inflection point. Historical patterns suggest that similar circumstances have led to transformative change, though outcomes remain highly dependent on coordinated action.',
  ];

  constructor(config: MockNewsConfig) {
    super();
    this.config = config;
  }

  /**
   * Get word category for contextual headlines
   */
  private getWordCategory(word: string): keyof typeof this.headlines {
    const upperWord = word.toUpperCase();
    const finance = ['BITCOIN', 'STOCKS', 'INFLATION', 'FEDERAL', 'MARKET', 'BANKING', 'CRYPTO', 'DEBT', 'NASDAQ', 'YIELD'];
    const politics = ['ELECTION', 'CONGRESS', 'SENATE', 'PRESIDENT', 'VOTE', 'POLL', 'CAMPAIGN', 'REFORM', 'BILL', 'POLICY'];
    const tech = ['AI', 'OPENAI', 'GOOGLE', 'APPLE', 'META', 'TESLA', 'CHIP', 'QUANTUM', 'ROBOT', 'CYBERSECURITY'];
    const climate = ['CLIMATE', 'CARBON', 'EMISSION', 'RENEWABLE', 'SOLAR', 'WIND', 'TEMPERATURE', 'DROUGHT', 'FLOOD', 'WILDFIRE'];
    const health = ['VACCINE', 'PANDEMIC', 'FDA', 'DRUG', 'HOSPITAL', 'MEDICARE', 'CANCER', 'TRIAL', 'OUTBREAK', 'VIRUS'];

    if (finance.includes(upperWord)) return 'finance';
    if (politics.includes(upperWord)) return 'politics';
    if (tech.includes(upperWord)) return 'tech';
    if (climate.includes(upperWord)) return 'climate';
    if (health.includes(upperWord)) return 'health';
    return 'general';
  }

  /**
   * Generate a professional headline using contextual templates
   */
  private generateHeadline(): string {
    // Pick primary word for headline context
    const primaryWord = this.config.wordPool[Math.floor(Math.random() * this.config.wordPool.length)];
    const category = this.getWordCategory(primaryWord);
    const categoryTemplates = this.headlines[category];
    const template = categoryTemplates[Math.floor(Math.random() * categoryTemplates.length)];

    // Replace first {WORD} with primary word
    let headline = template.replace('{WORD}', primaryWord);

    // Replace any additional {WORD} placeholders with related or random words
    while (headline.includes('{WORD}')) {
      const word = this.config.wordPool[Math.floor(Math.random() * this.config.wordPool.length)];
      headline = headline.replace('{WORD}', word);
    }

    return headline;
  }

  /**
   * Generate professional article content with multiple keyword mentions
   */
  private generateContent(): string {
    const numParagraphs = Math.floor(Math.random() * 2) + 1; // 1-2 paragraphs
    const content: string[] = [];

    for (let i = 0; i < numParagraphs; i++) {
      const template = this.contentTemplates[Math.floor(Math.random() * this.contentTemplates.length)];
      let paragraph = template;

      // Replace {WORD} placeholders with words from pool
      // Ensure we use multiple different words for variety
      const wordsUsed = new Set<string>();
      while (paragraph.includes('{WORD}')) {
        let word;
        let attempts = 0;
        // Try to pick a word we haven't used yet in this paragraph
        do {
          word = this.config.wordPool[Math.floor(Math.random() * this.config.wordPool.length)];
          attempts++;
        } while (wordsUsed.has(word) && attempts < 5);

        wordsUsed.add(word);
        paragraph = paragraph.replace('{WORD}', word);
      }

      content.push(paragraph);
    }

    return content.join(' ');
  }

  /**
   * Generate a complete news event
   */
  private generateEvent(): NewsEvent {
    const headline = this.generateHeadline();
    const content = this.generateContent();

    return {
      id: generateId(),
      timestamp: Date.now(),
      headline,
      content,
      matchedWords: [], // Will be filled by matching engine
      source: 'mock',
    };
  }

  /**
   * Schedule next event generation
   */
  private scheduleNext(): void {
    if (!this.isRunning) return;

    const delay = Math.random() *
      (this.config.maxInterval - this.config.minInterval) +
      this.config.minInterval;

    this.intervalId = setTimeout(() => {
      const event = this.generateEvent();
      this.notify(event);
      this.scheduleNext();
    }, delay);
  }

  /**
   * Start generating mock news
   */
  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    this.scheduleNext();
  }

  /**
   * Stop generating mock news
   */
  stop(): void {
    this.isRunning = false;

    if (this.intervalId) {
      clearTimeout(this.intervalId);
      this.intervalId = null;
    }
  }
}
