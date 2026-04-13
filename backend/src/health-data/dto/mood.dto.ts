import { IsNumber, IsString, IsEnum, IsOptional, IsDateString, Min, Max } from 'class-validator';
import { Type } from 'class-transformer';
import { Emotion } from '../schemas/mood-entry.schema';

export class MoodEntryDto {
  @IsEnum(Emotion)
  emotion: Emotion;

  @IsNumber()
  @Min(1)
  @Max(10)
  @Type(() => Number)
  stressLevel: number;

  @IsDateString()
  @IsOptional()
  timestamp?: string;

  @IsNumber()
  @IsOptional()
  @Type(() => Number)
  sleepHours?: number;

  @IsString()
  @IsOptional()
  notes?: string;
}

export class MoodHistoryQueryDto {
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
