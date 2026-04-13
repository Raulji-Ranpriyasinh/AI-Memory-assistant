import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type CgmReadingDocument = CgmReading & Document;

@Schema({ timestamps: true })
export class CgmReading {
  @Prop({ required: true })
  userId: string;

  @Prop({ required: true })
  glucoseMgDl: number;

  @Prop({ required: true, default: Date.now })
  timestamp: Date;

  @Prop({ type: String, enum: ['rising', 'falling', 'stable'] })
  trend?: string;

  @Prop()
  deviceId?: string;
}

export const CgmReadingSchema = SchemaFactory.createForClass(CgmReading);
