import { integer, jsonb, pgTable, real, serial, text, timestamp } from "drizzle-orm/pg-core";

export const runs = pgTable("runs", {
  id: serial("id").primaryKey(),
  modelId: text("model_id").notNull(),
  modelName: text("model_name").notNull(),
  taskId: text("task_id").notNull(),
  taskName: text("task_name").notNull(),
  config: jsonb("config").notNull(),
  params: integer("params").notNull(),
  activeParams: integer("active_params").notNull(),
  steps: integer("steps").notNull(),
  finalLoss: real("final_loss").notNull(),
  trainAcc: real("train_acc").notNull(),
  testAcc: real("test_acc").notNull(),
  durationMs: integer("duration_ms").notNull(),
  lossCurve: jsonb("loss_curve").notNull(),
  accCurve: jsonb("acc_curve").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type Run = typeof runs.$inferSelect;
export type NewRun = typeof runs.$inferInsert;
