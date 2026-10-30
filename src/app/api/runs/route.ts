import { db } from "@/db";
import { runs } from "@/db/schema";
import { desc } from "drizzle-orm";

export const dynamic = "force-dynamic";

export async function GET() {
  const rows = await db.select().from(runs).orderBy(desc(runs.createdAt)).limit(200);
  return Response.json(rows);
}

function num(v: unknown, d = 0): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

export async function POST(req: Request) {
  try {
    const b = await req.json();
    if (!b?.modelId || !b?.taskId) {
      return Response.json({ error: "modelId and taskId required" }, { status: 400 });
    }
    const [row] = await db
      .insert(runs)
      .values({
        modelId: String(b.modelId),
        modelName: String(b.modelName ?? b.modelId),
        taskId: String(b.taskId),
        taskName: String(b.taskName ?? b.taskId),
        config: b.config ?? {},
        params: Math.round(num(b.params)),
        activeParams: Math.round(num(b.activeParams)),
        steps: Math.round(num(b.steps)),
        finalLoss: num(b.finalLoss),
        trainAcc: num(b.trainAcc),
        testAcc: num(b.testAcc),
        durationMs: Math.round(num(b.durationMs)),
        lossCurve: Array.isArray(b.lossCurve) ? b.lossCurve : [],
        accCurve: Array.isArray(b.accCurve) ? b.accCurve : [],
      })
      .returning();
    return Response.json(row, { status: 201 });
  } catch (e) {
    return Response.json({ error: String(e) }, { status: 500 });
  }
}

export async function DELETE() {
  await db.delete(runs);
  return Response.json({ ok: true });
}
