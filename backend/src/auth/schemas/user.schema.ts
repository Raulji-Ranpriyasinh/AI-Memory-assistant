import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';
import * as mongoose from 'mongoose';

export enum UserRole {
  PATIENT = 'patient',
  PHYSICIAN = 'physician',
  ADMIN = 'admin',
}

export enum UserStatus {
  ACTIVE = 'active',
  SUSPENDED = 'suspended',
  PENDING = 'pending',
}

export type UserDocument = User & Document;

const ProfileSchema = new mongoose.Schema({
  firstName: String,
  lastName: String,
  dateOfBirth: Date,
  gender: String,
  phone: String,
  language: { type: String, default: 'en' },
  timezone: { type: String, default: 'UTC' },
}, { _id: false });

const HealthBaselineSchema = new mongoose.Schema({
  diabetesType: String,
  allergies: [String],
  chronicConditions: [String],
  currentMedications: [String],
  hba1c: Number,
  heightCm: Number,
  weightKg: Number,
}, { _id: false });

const PersonalityAssessmentSchema = new mongoose.Schema({
  completed: { type: Boolean, default: false },
  results: mongoose.Schema.Types.Mixed,
  completedAt: Date,
}, { _id: false });

const ConsentSchema = new mongoose.Schema({
  dataProcessing: Boolean,
  healthDataSharing: Boolean,
  marketingEmails: Boolean,
  consentedAt: Date,
}, { _id: false });

const NotificationPreferencesSchema = new mongoose.Schema({
  push: { type: Boolean, default: true },
  email: { type: Boolean, default: true },
  sms: { type: Boolean, default: false },
  glucoseAlerts: { type: Boolean, default: true },
  mealReminders: { type: Boolean, default: true },
  medicationReminders: { type: Boolean, default: true },
}, { _id: false });

@Schema({ timestamps: true })
export class User {
  @Prop({ required: true, unique: true, lowercase: true, trim: true })
  email: string;

  @Prop({ required: true })
  passwordHash: string;

  @Prop({ type: String, enum: UserRole, default: UserRole.PATIENT })
  role: UserRole;

  @Prop({ type: String, enum: UserStatus, default: UserStatus.PENDING })
  status: UserStatus;

  @Prop({ type: ProfileSchema, default: {} })
  profile: Record<string, any>;

  @Prop({ type: HealthBaselineSchema, default: {} })
  healthBaseline: Record<string, any>;

  @Prop({ type: PersonalityAssessmentSchema, default: {} })
  personalityAssessment: Record<string, any>;

  @Prop({ type: ConsentSchema, default: {} })
  consent: Record<string, any>;

  @Prop({ type: NotificationPreferencesSchema, default: {} })
  notificationPreferences: Record<string, any>;

  @Prop()
  linkedClinicId?: string;

  @Prop()
  linkedPhysicianId?: string;

  @Prop()
  stripeCustomerId?: string;

  @Prop({ default: Date.now })
  createdAt: Date;

  @Prop({ default: Date.now })
  updatedAt: Date;
}

export const UserSchema = SchemaFactory.createForClass(User);
