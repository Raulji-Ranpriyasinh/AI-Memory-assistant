import { IsNumber, IsDateString, IsOptional, IsString, IsEnum, IsArray } from 'class-validator';
import { Type } from 'class-transformer';

export class CgmReadingDto {
  @IsNumber()
  glucoseMgDl: number;

  @IsDateString()
  @IsOptional()
  timestamp?: string;

  @IsEnum(['rising', 'falling', 'stable'])
  @IsOptional()
  trend?: string;

  @IsString()
  @IsOptional()
  deviceId?: string;
}

export class BulkCgmReadingsDto {
  @IsArray()
  readings: CgmReadingDto[];
}

export class CgmSummaryQueryDto {
  @IsOptional()
  @IsString()
  period?: string;

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
