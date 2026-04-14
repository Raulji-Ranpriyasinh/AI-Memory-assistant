import { IsNumber, IsString, IsEnum, IsOptional, IsDateString, IsArray } from 'class-validator';
import { Type } from 'class-transformer';
import { MealType, GlycemicLoad } from '../schemas/food-log.schema';

export class FoodLogDto {
  @IsEnum(MealType)
  mealType: MealType;

  @IsArray()
  @IsString({ each: true })
  items: string[];

  @IsDateString()
  @IsOptional()
  timestamp?: string;

  @IsNumber()
  @IsOptional()
  @Type(() => Number)
  estimatedCalories?: number;

  @IsEnum(GlycemicLoad)
  @IsOptional()
  glycemicLoad?: GlycemicLoad;

  @IsString()
  @IsOptional()
  photoUrl?: string;

  @IsString()
  @IsOptional()
  imageBase64?: string;
}

export class FoodRecognizeDto {
  @IsString()
  imageBase64: string;
}

export class FoodHistoryQueryDto {
  @IsOptional()
  @IsDateString()
  from?: string;

  @IsOptional()
  @IsDateString()
  to?: string;

  @IsOptional()
  @IsNumber()
  @Type(() => Number)
  limit?: number;
}
