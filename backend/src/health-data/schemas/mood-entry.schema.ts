import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export enum Emotion {
  HAPPY = 'happy',
  SAD = 'sad',
  ANGRY = 'angry',
  ANXIOUS = 'anxious',
  TIRED = 'tired',
  CALM = 'calm',
  EXCITED = 'excited',
  FRUSTRATED = 'frustrated',
  NEUTRAL = 'neutral',
}

export type MoodEntryDocument = MoodEntry & Document;

@Schema({ timestamps: true })
export class MoodEntry {
  @Prop({ required: true })
  userId: string;

  @Prop({ required: true, type: String, enum: Object.values(Emotion) })
  emotion: Emotion;

  @Prop({ required: true, min: 1, max: 10 })
  stressLevel: number;

  @Prop({ required: true, default: Date.now })
  timestamp: Date;

  @Prop()
  sleepHours?: number;

  @Prop()
  notes?: string;
}

export const MoodEntrySchema = SchemaFactory.createForClass(MoodEntry);
