import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export enum MealType {
  BREAKFAST = 'breakfast',
  LUNCH = 'lunch',
  DINNER = 'dinner',
  SNACK = 'snack',
}

export enum GlycemicLoad {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
}

export type FoodLogDocument = FoodLog & Document;

@Schema({ timestamps: true })
export class FoodLog {
  @Prop({ required: true })
  userId: string;

  @Prop({ required: true, type: String, enum: Object.values(MealType) })
  mealType: MealType;

  @Prop({ type: [String], required: true })
  items: string[];

  @Prop({ required: true, default: Date.now })
  timestamp: Date;

  @Prop()
  estimatedCalories?: number;

  @Prop({ type: String, enum: Object.values(GlycemicLoad) })
  glycemicLoad?: GlycemicLoad;

  @Prop()
  photoUrl?: string;

  @Prop()
  imageBase64?: string;
}

export const FoodLogSchema = SchemaFactory.createForClass(FoodLog);
