import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export enum Intensity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
}

export type ActivityLogDocument = ActivityLog & Document;

@Schema({ timestamps: true })
export class ActivityLog {
  @Prop({ required: true })
  userId: string;

  @Prop({ required: true })
  activityType: string;

  @Prop({ required: true })
  durationMinutes: number;

  @Prop({ required: true, default: Date.now })
  timestamp: Date;

  @Prop({ type: String, enum: Object.values(Intensity) })
  intensity?: Intensity;
}

export const ActivityLogSchema = SchemaFactory.createForClass(ActivityLog);
