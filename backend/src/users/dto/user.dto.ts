import { IsOptional, IsString, IsEnum, IsDateString, IsNumber, IsArray, IsBoolean, ValidateNested } from 'class-validator';
import { Type } from 'class-transformer';

export class UpdateProfileDto {
  @IsOptional()
  @IsString()
  firstName?: string;

  @IsOptional()
  @IsString()
  lastName?: string;

  @IsOptional()
  @IsDateString()
  dateOfBirth?: Date;

  @IsOptional()
  @IsString()
  gender?: string;

  @IsOptional()
  @IsString()
  phone?: string;

  @IsOptional()
  @IsString()
  language?: string;

  @IsOptional()
  @IsString()
  timezone?: string;
}

export class HealthBaselineDto {
  @IsOptional()
  @IsString()
  diabetesType?: string;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  allergies?: string[];

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  chronicConditions?: string[];

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  currentMedications?: string[];

  @IsOptional()
  @IsNumber()
  hba1c?: number;

  @IsOptional()
  @IsNumber()
  heightCm?: number;

  @IsOptional()
  @IsNumber()
  weightKg?: number;
}

export class PersonalityAssessmentDto {
  @IsOptional()
  results?: Record<string, any>;
}

export class UpdateConsentDto {
  @IsOptional()
  @IsBoolean()
  dataProcessing?: boolean;

  @IsOptional()
  @IsBoolean()
  healthDataSharing?: boolean;

  @IsOptional()
  @IsBoolean()
  marketingEmails?: boolean;
}

export class UpdateNotificationPreferencesDto {
  @IsOptional()
  @IsBoolean()
  push?: boolean;

  @IsOptional()
  @IsBoolean()
  email?: boolean;

  @IsOptional()
  @IsBoolean()
  sms?: boolean;

  @IsOptional()
  @IsBoolean()
  glucoseAlerts?: boolean;

  @IsOptional()
  @IsBoolean()
  mealReminders?: boolean;

  @IsOptional()
  @IsBoolean()
  medicationReminders?: boolean;
}
