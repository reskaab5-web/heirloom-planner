require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { Pool } = require('pg');
const https = require('https');
const Anthropic = require('@anthropic-ai/sdk');
const rateLimit = require('express-rate-limit');

const app = express();
const PORT = process.env.PORT || 3001;
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || 'http://localhost:3000';
const API_KEY = process.env.API_KEY;
const ANALYSIS_MODEL = process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-6';
const MAX_BATCH = 25;

// Dissents cost 2 extra API calls per case. CourtListener's default limits are
// 5/min, 50/hour, 125/day — so a 25-case collect with dissents is ~75 calls and
// will blow the hourly cap. Leave off until you're on bulk data or a partnership.
const COLLECT_DISSENTS = process.env.COLLECT_DISSENTS === 'true';

// Characters of opinion text sent to the model. Store everything; decide here
// what fits the context and the bill.
const ANALYSIS_CHARS = Number(process.env.ANALYSIS_CHARS || 24000);

if (!API_KEY) {
    console.warn('⚠️  API_KEY not set — POST endpoints will reject all requests.');
}

app.use(cors({ origin: ALLOWED_ORIGIN, credentials: false }));
app.use(express.json({ limit: '100kb' }));

const db = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.DATABASE_URL && process.env.DATABASE_URL.includes('supabase')
        ? { rejectUnauthorized: false }
        : false
});

const client = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY
});

function requireApiKey(req, res, next) {
    if (!API_KEY || req.headers['x-api-key'] !== API_KEY) {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    next();
}

function serverError(res, error, label) {
    console.error(label, error);
    return res.status(500).json({ error: 'Internal server error' });
}

const globalLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 200,
    standardHeaders: true,
    legacyHeaders: false
});

const expensiveLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 10,
    standardHeaders: true,
    legacyHeaders: false
});

app.use('/api/', globalLimiter);

db.query('SELECT NOW()', (err, result) => {
    if (err) {
        console.error('❌ Database connection error:', err);
    } else {
        console.log('✅ Database connected');
    }
});

class CourtListenerAPI {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.rateLimitDelay = 12000;
    }

    async makeRequest(path) {
        return new Promise((resolve, reject) => {
            const options = {
                hostname: 'www.courtlistener.com',
                path: `/api/rest/v4${path}`,
                method: 'GET',
                headers: {
                    'Authorization': `Token ${this.apiKey}`,
                    'User-Agent': 'CourtIntelligence/1.0',
                    'Accept': 'application/json'
                }
            };

            https.get(options, (res) => {
                let data = '';
                res.on('data', (chunk) => { data += chunk; });
                res.on('end', () => {
                    if (res.statusCode === 401) {
                        return reject(new Error('401 Unauthorized - check COURTLISTENER_API_KEY'));
                    }
                    try {
                        resolve(JSON.parse(data));
                    } catch (e) {
                        reject(new Error(`Failed to parse: ${e.message}`));
                    }
                });
            }).on('error', reject);
        });
    }

    async delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async search({ query = '', courtId = '', limit = 10 } = {}) {
        const params = new URLSearchParams();
        params.set('type', 'o');
        if (query) params.set('q', query);
        if (courtId) params.set('court', courtId);
        params.set('order_by', 'dateFiled desc');

        const response = await this.makeRequest(`/search/?${params.toString()}`);
        await this.delay(this.rateLimitDelay);

        if (response.results && Array.isArray(response.results)) {
            return response.results.slice(0, limit);
        }
        return [];
    }

    /**
     * Fetch a single opinion by its OPINION id.
     *
     * NOTE: opinion ids and cluster ids are different namespaces. Passing a
     * cluster id here silently returns a different case. Palsgraf is cluster
     * 3602780 / opinion 3584293; fetching opinion 3602780 returns Flynn v.
     * Equitable Life. Search results carry `opinion_id` — use that.
     */
    async getOpinionById(opinionId) {
        try {
            const response = await this.makeRequest(`/opinions/${opinionId}/?format=json`);
            await this.delay(this.rateLimitDelay);
            return response;
        } catch (error) {
            console.error(`Error fetching opinion ${opinionId}:`, error.message);
            return null;
        }
    }

    /**
     * Resolve a cluster id to its opinion ids. Needed when repairing rows
     * stored before opinion_id was recorded — absolute_url contains the
     * CLUSTER id, not the opinion id.
     */
    async getClusterOpinionIds(clusterId) {
        try {
            const cluster = await this.makeRequest(`/clusters/${clusterId}/?format=json`);
            await this.delay(this.rateLimitDelay);
            const subs = Array.isArray(cluster?.sub_opinions) ? cluster.sub_opinions : [];
            // sub_opinions come back as absolute URLs ending in /{id}/
            return subs
                .map((u) => {
                    const m = String(u).match(/\/(\d+)\/?$/);
                    return m ? Number(m[1]) : null;
                })
                .filter(Boolean);
        } catch (error) {
            console.error(`Error resolving cluster ${clusterId}:`, error.message);
            return [];
        }
    }
}

/**
 * Pull readable text out of an opinion record.
 *
 * `plain_text` is empty on a large share of the corpus — the populated field
 * is usually `html_with_citations`. Reading only plain_text is why cases sit
 * on needs_analysis forever: /api/analyze-all filters on opinion_text != ''.
 */
function extractOpinionText(opinion) {
    if (!opinion) return '';
    const raw =
        opinion.html_with_citations ||
        opinion.plain_text ||
        opinion.html ||
        opinion.html_lawbox ||
        opinion.html_columbia ||
        opinion.xml_harvard ||
        '';
    return String(raw)
        .replace(/<[^>]+>/g, ' ')
        .replace(/&nbsp;/g, ' ')
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/\s+/g, ' ')
        .trim();
}

async function initializeDatabase() {
    try {
        await db.query(`
            CREATE TABLE IF NOT EXISTS cases (
                id VARCHAR(500) PRIMARY KEY,
                name VARCHAR(500) NOT NULL,
                docket VARCHAR(500),
                court VARCHAR(500),
                court_id VARCHAR(100),
                citation TEXT,
                date_decided TIMESTAMP,
                opinion_text TEXT,
                judges JSONB,
                summary TEXT,
                pro_arguments JSONB,
                con_arguments JSONB,
                impact TEXT,
                topics JSONB,
                primary_topic VARCHAR(200),
                topic_confidence VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                needs_analysis BOOLEAN DEFAULT true
            );
        `);
        await db.query(`ALTER TABLE cases ADD COLUMN IF NOT EXISTS court_id VARCHAR(100);`);
        await db.query(`ALTER TABLE cases ADD COLUMN IF NOT EXISTS citation TEXT;`);
        await db.query(`ALTER TABLE cases ADD COLUMN IF NOT EXISTS primary_topic VARCHAR(200);`);
        await db.query(`ALTER TABLE cases ADD COLUMN IF NOT EXISTS topic_confidence VARCHAR(20);`);
        // Recording these makes future refetches exact instead of guesswork.
        await db.query(`ALTER TABLE cases ADD COLUMN IF NOT EXISTS opinion_id BIGINT;`);
        await db.query(`ALTER TABLE cases ADD COLUMN IF NOT EXISTS cluster_id BIGINT;`);
        await db.query(`ALTER TABLE cases ADD COLUMN IF NOT EXISTS dissent_text TEXT;`);
        return true;
    } catch (error) {
        console.error('Database init error:', error.message);
        return false;
    }
}

async function caseExists(caseId) {
    try {
        const result = await db.query('SELECT id FROM cases WHERE id = $1', [caseId]);
        return result.rows.length > 0;
    } catch (error) {
        return false;
    }
}

async function storeCase(r, opinionText = '', dissentText = '') {
    const caseId = r.absolute_url || `case-${r.cluster_id}`;
    const citation = Array.isArray(r.citation) ? r.citation.join('; ') : (r.citation || '');
    try {
        await db.query(
            `INSERT INTO cases
                (id, name, docket, court, court_id, citation, date_decided, opinion_text,
                 dissent_text, summary, opinion_id, cluster_id, needs_analysis)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, true)
             ON CONFLICT (id) DO UPDATE SET
                opinion_text = EXCLUDED.opinion_text,
                dissent_text = EXCLUDED.dissent_text,
                summary = EXCLUDED.summary,
                opinion_id = EXCLUDED.opinion_id,
                cluster_id = EXCLUDED.cluster_id,
                needs_analysis = true`,
            [
                caseId,
                (r.caseName || 'Unknown').substring(0, 500),
                (r.docketNumber || '').substring(0, 500),
                (r.court || '').substring(0, 500),
                (r.court_id || '').substring(0, 100),
                citation,
                r.dateFiled || null,
                // Store the full opinion. Truncating at 5k threw away ~94% of a
                // major case — Twombly alone is ~70k characters. Chunk at prompt
                // time instead, where you can choose what to keep.
                opinionText || '',
                dissentText || '',
                (r.summary || '').substring(0, 2000) || '',
                r.opinion_id || null,
                r.cluster_id || null
            ]
        );
        return true;
    } catch (error) {
        console.error('Store error:', error.message);
        return false;
    }
}

const safeParse = (val) => {
    if (val == null) return [];
    if (Array.isArray(val)) return val;
    if (typeof val === 'object') return val;
    try {
        return JSON.parse(val);
    } catch {
        return [val];
    }
};

const normalizeCase = (c) => ({
    ...c,
    proArguments: safeParse(c.proArguments),
    conArguments: safeParse(c.conArguments),
    topics: safeParse(c.topics)
});

const buildLegalAnalysisPrompt = (caseName, summary, opinionText, dissentText = '') => `
You are an expert legal analyst preparing case intelligence for law firm research.

Analyze this court case and provide STRUCTURED LEGAL RESEARCH INSIGHTS:

CASE: ${caseName}
${summary ? `SUMMARY: ${summary}` : ''}
OPINION: ${String(opinionText).substring(0, ANALYSIS_CHARS)}
${dissentText ? `\nDISSENT/CONCURRENCE: ${String(dissentText).substring(0, 8000)}` : ''}

Respond ONLY in this JSON format, no other text:
{
  "coreHolding": "The definitive ruling in 1-2 sentences that establishes legal precedent",
  
  "legalPrinciples": [
    "Specific legal principle #1 established by this case",
    "Specific legal principle #2 established by this case",
    "Specific legal principle #3 established by this case"
  ],
  
  "proceduralContext": {
    "courtLevel": "Trial/Appellate/Supreme Court",
    "procedureAtIssue": "What procedural rule or issue was central",
    "standard": "Standard of review applied (de novo, abuse of discretion, etc.)"
  },
  
  "substantiveIssue": {
    "area": "Practice area (Constitutional, Contract, Employment, IP, Corporate, etc.)",
    "keyFacts": "2-3 critical facts that distinguish this case",
    "legalQuestion": "The specific legal question answered"
  },
  
  "precedentAnalysis": {
    "overrules": "Any prior cases explicitly overruled or limited",
    "affirms": "What prior law this reinforces or clarifies",
    "distinguishes": "How this case narrows or expands prior precedent"
  },
  
  "jurisdictionalImpact": {
    "jurisdiction": "Jurisdiction (Federal, State, Circuit, etc.)",
    "bindingEffect": "Who is bound by this ruling and why",
    "applicableStatutes": "Statutes or regulations central to the holding",
    "practiceImpact": "How this affects legal practice in this jurisdiction"
  },
  
  "practiceApplications": [
    "Specific application for employment lawyers",
    "Specific application for corporate/contract lawyers",
    "Specific application for civil rights lawyers",
    "Specific application for litigation strategy"
  ],
  
  "dissent": {
    "exists": true or false,
    "mainArgument": "If dissent exists, what was the key disagreement",
    "significance": "Why the dissent matters (potential future reversal, etc.)"
  },
  
  "litigationImplications": {
    "favorableForParty": "Which party type benefits from this ruling",
    "riskFactors": "Fact patterns that could distinguish cases unfavorably",
    "appealPotential": "Likelihood and grounds for appeal/reversal"
  },
  
  "researchNotes": [
    "Note 1: Cases that cite this decision frequently",
    "Note 2: Practical guidance for lawyers using this precedent",
    "Note 3: Potential legislative response or statutory amendments"
  ]
}
`;

app.get('/api/cases', async (req, res) => {
    try {
        const result = await db.query(`
            SELECT 
                id, name, docket, court, date_decided as "dateFiled",
                summary, pro_arguments as "proArguments",
                con_arguments as "conArguments", impact, topics,
                primary_topic as "primaryTopic",
                topic_confidence as "topicConfidence"
            FROM cases
            ORDER BY date_decided DESC
        `);

        const cases = result.rows.map(normalizeCase);

        res.json(cases);
    } catch (error) {
        console.error('Error fetching cases:', error);
        res.status(500).json({ error: 'Failed to fetch cases' });
    }
});

app.get('/api/topics', async (req, res) => {
    try {
        const result = await db.query(`
            SELECT DISTINCT court, COUNT(*) as count
            FROM cases
            WHERE court IS NOT NULL AND court != ''
            GROUP BY court
            ORDER BY count DESC
        `);
        res.json(result.rows);
    } catch (error) {
        console.error('Error fetching topics:', error);
        res.status(500).json({ error: 'Failed to fetch topics' });
    }
});

app.get('/api/search', async (req, res) => {
    try {
        const rawQ = typeof req.query.q === 'string' ? req.query.q : '';
        const q = rawQ.replace(/[%_\\]/g, '\\$&');
        if (!q || q.length < 2) {
            return res.json([]);
        }

        const result = await db.query(`
            SELECT 
                id, name, docket, court, date_decided as "dateFiled",
                summary, pro_arguments as "proArguments",
                con_arguments as "conArguments", impact, topics,
                primary_topic as "primaryTopic",
                topic_confidence as "topicConfidence"
            FROM cases
            WHERE (name ILIKE $1 OR docket ILIKE $1 OR summary ILIKE $1 OR court ILIKE $1)
            ORDER BY date_decided DESC
            LIMIT 20
        `, ['%' + q + '%']);

        const cases = result.rows.map(normalizeCase);

        res.json(cases);
    } catch (error) {
        console.error('Error searching:', error);
        res.status(500).json({ error: 'Failed to search' });
    }
});

app.post('/api/collect', requireApiKey, expensiveLimiter, async (req, res) => {
    try {
        const query = typeof req.body.query === 'string' ? req.body.query.slice(0, 500) : '';
        const courtId = typeof req.body.courtId === 'string' ? req.body.courtId.slice(0, 100) : '';
        const limit = Math.min(Math.max(parseInt(req.body.limit, 10) || 10, 1), MAX_BATCH);

        if (!process.env.COURTLISTENER_API_KEY) {
            return res.status(400).json({ 
                success: false, 
                error: 'COURTLISTENER_API_KEY not configured' 
            });
        }

        const dbReady = await initializeDatabase();
        if (!dbReady) {
            return res.status(500).json({ 
                success: false, 
                error: 'Failed to initialize database' 
            });
        }

        console.log(`\n📥 Collection started: query="${query}" court="${courtId}" limit=${limit}`);

        const api = new CourtListenerAPI(process.env.COURTLISTENER_API_KEY);
        const results = await api.search({ query, courtId, limit });

        if (!results || results.length === 0) {
            return res.json({
                success: true,
                newCases: 0,
                totalFetched: 0,
                cases: [],
                message: 'No cases found matching criteria'
            });
        }

        let newCases = 0;
        const storedCases = [];

        for (const r of results) {
            try {
                const caseId = r.absolute_url || `case-${r.cluster_id}`;
                const isNew = !(await caseExists(caseId));

                let opinionText = '';
                let dissentText = '';

                // Use opinion_id, NOT cluster_id. Different namespaces.
                if (r.opinion_id) {
                    const opinion = await api.getOpinionById(r.opinion_id);
                    opinionText = extractOpinionText(opinion);
                } else if (r.cluster_id) {
                    // Older/odd results without opinion_id: resolve via the cluster.
                    const ids = await api.getClusterOpinionIds(r.cluster_id);
                    if (ids.length) {
                        const opinion = await api.getOpinionById(ids[0]);
                        opinionText = extractOpinionText(opinion);
                    }
                }

                // A dissent is the thing students get cold-called on, and the
                // analysis schema already has a `dissent` field to fill.
                if (COLLECT_DISSENTS && r.cluster_id) {
                    const ids = await api.getClusterOpinionIds(r.cluster_id);
                    const others = ids.filter((id) => id !== r.opinion_id);
                    if (others.length) {
                        const sib = await api.getOpinionById(others[0]);
                        if (sib && sib.type && /dissent|concur/i.test(sib.type)) {
                            dissentText = extractOpinionText(sib);
                        }
                    }
                }

                if (!opinionText) {
                    console.warn(`⚠️  No opinion text for ${r.caseName} (opinion_id=${r.opinion_id})`);
                }

                const stored = await storeCase(r, opinionText, dissentText);
                if (stored && isNew) {
                    newCases++;
                }
                
                storedCases.push({
                    name: r.caseName,
                    docket: r.docketNumber || 'n/a',
                    court: r.court || 'n/a',
                    citation: r.citation,
                    dateFiled: r.dateFiled,
                    opinionLength: opinionText.length
                });
            } catch (error) {
                console.error(`Error processing case:`, error.message);
            }
        }

        console.log(`✅ Collection complete: ${newCases} new cases stored (${results.length} total fetched)\n`);

        res.json({
            success: true,
            newCases,
            totalFetched: results.length,
            cases: storedCases,
            message: `Stored ${newCases} new case(s) with opinion text`
        });

    } catch (error) {
        return serverError(res, error, 'Collection error:');
    }
});

app.post('/api/analyze-case', requireApiKey, expensiveLimiter, async (req, res) => {
    try {
        const caseId = typeof req.body.caseId === 'string' ? req.body.caseId : '';
        const caseName = typeof req.body.caseName === 'string' ? req.body.caseName : '';
        const summary = typeof req.body.summary === 'string' ? req.body.summary : '';
        const opinionText = typeof req.body.opinionText === 'string'
            ? req.body.opinionText.slice(0, 20000) : '';

        if (!caseId || !caseName || !opinionText) {
            return res.status(400).json({
                error: "Missing required fields: caseId, caseName, opinionText",
            });
        }

        if (!process.env.ANTHROPIC_API_KEY) {
            return res.status(400).json({
                error: "ANTHROPIC_API_KEY not configured"
            });
        }

        const prompt = buildLegalAnalysisPrompt(caseName, summary, opinionText);

        const message = await client.messages.create({
            model: ANALYSIS_MODEL,
            max_tokens: 2048,
            messages: [
                {
                    role: "user",
                    content: prompt,
                },
            ],
        });

        const responseText = message.content[0].type === "text" ? message.content[0].text : "";

        let analysis;
        try {
            const jsonMatch = responseText.match(/\{[\s\S]*\}/);
            if (!jsonMatch) {
                throw new Error("No JSON found in response");
            }
            analysis = JSON.parse(jsonMatch[0]);
        } catch (parseError) {
            console.error("Failed to parse Claude response:", responseText);
            return res.status(502).json({ error: "Failed to parse analysis response" });
        }

        const updateQuery = `
            UPDATE cases 
            SET 
                pro_arguments = $1,
                con_arguments = $2,
                impact = $3,
                needs_analysis = false
            WHERE id = $4
            RETURNING *;
        `;

        const result = await db.query(updateQuery, [
            JSON.stringify(analysis),
            JSON.stringify({
                dissent: analysis.dissent,
                riskFactors: analysis.litigationImplications?.riskFactors
            }),
            JSON.stringify({
                coreHolding: analysis.coreHolding,
                practiceImpact: analysis.jurisdictionalImpact?.practiceImpact,
                precedentEffect: analysis.precedentAnalysis
            }),
            caseId,
        ]);

        if (result.rows.length === 0) {
            return res.status(404).json({ error: "Case not found" });
        }

        res.json({
            success: true,
            message: "Case analysis completed with legal research insights",
            case: result.rows[0],
            analysis: analysis,
        });
    } catch (error) {
        return serverError(res, error, 'Analysis endpoint error:');
    }
});

app.post("/api/analyze-all", requireApiKey, expensiveLimiter, async (req, res) => {
    try {
        const limit = Math.min(Math.max(parseInt(req.body.limit, 10) || 10, 1), MAX_BATCH);

        if (!process.env.ANTHROPIC_API_KEY) {
            return res.status(400).json({
                error: "ANTHROPIC_API_KEY not configured"
            });
        }

        const casesQuery = `
            SELECT id, name, summary, opinion_text 
            FROM cases 
            WHERE needs_analysis = true AND opinion_text IS NOT NULL AND opinion_text != ''
            LIMIT $1;
        `;

        const casesResult = await db.query(casesQuery, [limit]);
        const cases = casesResult.rows;

        if (cases.length === 0) {
            return res.json({
                message: "No cases with opinion text need analysis",
                analyzed: 0,
            });
        }

        const analyzed = [];
        const failed = [];

        console.log(`\n📋 Starting legal analysis of ${cases.length} cases...\n`);

        for (const caseItem of cases) {
            try {
                console.log(`⏳ Analyzing: ${caseItem.name}`);

                const prompt = buildLegalAnalysisPrompt(caseItem.name, caseItem.summary, caseItem.opinion_text);

                const message = await client.messages.create({
                    model: ANALYSIS_MODEL,
                    max_tokens: 2048,
                    messages: [
                        {
                            role: "user",
                            content: prompt,
                        },
                    ],
                });

                const responseText = message.content[0].type === "text" ? message.content[0].text : "";

                const jsonMatch = responseText.match(/\{[\s\S]*\}/);
                if (!jsonMatch) {
                    throw new Error("No JSON in response");
                }
                const analysis = JSON.parse(jsonMatch[0]);

                const updateQuery = `
                    UPDATE cases 
                    SET 
                        pro_arguments = $1,
                        con_arguments = $2,
                        impact = $3,
                        needs_analysis = false
                    WHERE id = $4;
                `;

                await db.query(updateQuery, [
                    JSON.stringify(analysis),
                    JSON.stringify({
                        dissent: analysis.dissent,
                        riskFactors: analysis.litigationImplications?.riskFactors
                    }),
                    JSON.stringify({
                        coreHolding: analysis.coreHolding,
                        practiceImpact: analysis.jurisdictionalImpact?.practiceImpact,
                        precedentEffect: analysis.precedentAnalysis
                    }),
                    caseItem.id,
                ]);

                analyzed.push({
                    id: caseItem.id,
                    name: caseItem.name,
                    area: analysis.substantiveIssue?.area,
                    holding: analysis.coreHolding,
                    status: "success",
                });

                console.log(`✅ Analyzed: ${caseItem.name}`);

                await new Promise((resolve) => setTimeout(resolve, 500));
            } catch (error) {
                failed.push({
                    id: caseItem.id,
                    name: caseItem.name,
                    error: error.message,
                });
                console.error(`❌ Failed to analyze ${caseItem.name}:`, error.message);
            }
        }

        console.log(`\n✨ Legal analysis batch complete!\n`);

        res.json({
            success: true,
            message: `Legal analysis batch complete`,
            analyzed: analyzed.length,
            failed: failed.length,
            details: {
                successful: analyzed,
                failed: failed,
            },
        });
    } catch (error) {
        return serverError(res, error, 'Batch analysis error:');
    }
});

/**
 * Repair rows stored before the opinion-id fix.
 *
 * Any case collected previously has opinion_text belonging to a DIFFERENT case
 * (or is empty), because getOpinion() was called with a cluster id. This
 * re-resolves each stored case through its cluster and refetches the correct
 * opinion, then marks it for re-analysis.
 *
 * Rate limits are the binding constraint: 5/min, 50/hour, 125/day. Each case
 * costs 2 calls (cluster + opinion), so keep `limit` small and run it
 * repeatedly rather than trying to fix everything at once.
 *
 *   curl -X POST $API/api/repair -H "x-api-key: $API_KEY" \
 *        -H 'content-type: application/json' -d '{"limit":10}'
 */
app.post('/api/repair', requireApiKey, expensiveLimiter, async (req, res) => {
    try {
        const limit = Math.min(Math.max(parseInt(req.body.limit, 10) || 10, 1), MAX_BATCH);

        if (!process.env.COURTLISTENER_API_KEY) {
            return res.status(400).json({ error: 'COURTLISTENER_API_KEY not configured' });
        }
        await initializeDatabase();

        // Prefer rows never repaired (no opinion_id recorded) — those are the
        // ones written by the buggy path.
        const { rows } = await db.query(
            `SELECT id, name, cluster_id, opinion_id
               FROM cases
              WHERE opinion_id IS NULL
              ORDER BY created_at ASC
              LIMIT $1`,
            [limit]
        );

        if (rows.length === 0) {
            return res.json({ success: true, repaired: 0, message: 'Nothing left to repair.' });
        }

        const api = new CourtListenerAPI(process.env.COURTLISTENER_API_KEY);
        const repaired = [];
        const failed = [];

        for (const row of rows) {
            try {
                // absolute_url looks like /opinion/{cluster_id}/{slug}/
                const m = String(row.id).match(/\/opinion\/(\d+)\//);
                const clusterId = row.cluster_id || (m ? Number(m[1]) : null);
                if (!clusterId) {
                    failed.push({ id: row.id, name: row.name, error: 'no cluster id' });
                    continue;
                }

                const ids = await api.getClusterOpinionIds(clusterId);
                if (!ids.length) {
                    failed.push({ id: row.id, name: row.name, error: 'cluster has no opinions' });
                    continue;
                }

                const opinion = await api.getOpinionById(ids[0]);
                const text = extractOpinionText(opinion);
                if (!text) {
                    failed.push({ id: row.id, name: row.name, error: 'opinion had no text' });
                    continue;
                }

                await db.query(
                    `UPDATE cases
                        SET opinion_text = $1,
                            opinion_id = $2,
                            cluster_id = $3,
                            needs_analysis = true
                      WHERE id = $4`,
                    [text, ids[0], clusterId, row.id]
                );

                repaired.push({ id: row.id, name: row.name, opinion_id: ids[0], chars: text.length });
                console.log(`🔧 Repaired ${row.name} → opinion ${ids[0]} (${text.length} chars)`);
            } catch (error) {
                failed.push({ id: row.id, name: row.name, error: error.message });
            }
        }

        const remaining = await db.query(`SELECT COUNT(*)::int AS n FROM cases WHERE opinion_id IS NULL`);

        res.json({
            success: true,
            repaired: repaired.length,
            failed: failed.length,
            remaining: remaining.rows[0].n,
            details: { repaired, failed },
            message: repaired.length
                ? `Repaired ${repaired.length}. Re-run /api/analyze-all to redo the analysis on corrected text.`
                : 'No rows could be repaired in this batch.'
        });
    } catch (error) {
        return serverError(res, error, 'Repair endpoint error:');
    }
});

app.get("/api/analysis-status", async (req, res) => {
    try {
        const result = await db.query(`
            SELECT 
                COUNT(*) as total_cases,
                SUM(CASE WHEN needs_analysis = true THEN 1 ELSE 0 END) as needs_analysis,
                SUM(CASE WHEN needs_analysis = false THEN 1 ELSE 0 END) as analyzed
            FROM cases;
        `);

        res.json(result.rows[0]);
    } catch (error) {
        return serverError(res, error, 'Analysis status error:');
    }
});

app.post('/api/delete-all', requireApiKey, async (req, res) => {
    try {
        await db.query('DELETE FROM cases');
        res.json({ success: true, message: 'All cases deleted' });
    } catch (error) {
        return serverError(res, error, 'Delete-all error:');
    }
});

app.get('/health', (req, res) => {
    res.json({ status: 'ok', message: 'Federal Courts Intelligence API' });
});

app.listen(PORT, () => {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`🏛️  Federal Courts Intelligence API`);
    console.log(`${'='.repeat(60)}`);
    console.log(`✅ Server running on http://localhost:${PORT}`);
    console.log(`📊 API: http://localhost:${PORT}/api/cases`);
    console.log(`🔍 Search: http://localhost:${PORT}/api/search?q=...`);
    console.log(`📥 Collect: POST http://localhost:${PORT}/api/collect`);
    console.log(`🤖 Analyze: POST http://localhost:${PORT}/api/analyze-all`);
    console.log(`📊 Status: http://localhost:${PORT}/api/analysis-status`);
    console.log(`🗑️  Delete: POST http://localhost:${PORT}/api/delete-all`);
    console.log(`💚 Health: http://localhost:${PORT}/health`);
    console.log(`${'='.repeat(60)}\n`);
});

module.exports = app;