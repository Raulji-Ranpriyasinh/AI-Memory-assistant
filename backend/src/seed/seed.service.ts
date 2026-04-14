import { Injectable, OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import * as bcrypt from 'bcrypt';
import { User, UserDocument, UserRole, UserStatus } from '../auth/schemas/user.schema';

@Injectable()
export class SeedService implements OnModuleInit {
  constructor(
    @InjectModel(User.name) private userModel: Model<UserDocument>,
  ) {}

  async onModuleInit() {
    await this.seedAdminUser();
    await this.seedDemoUser();
  }

  private async seedAdminUser(): Promise<void> {
    const adminEmail = 'admin@delight.health';
    const existingAdmin = await this.userModel.findOne({ email: adminEmail });

    if (existingAdmin) {
      console.log('[SEED] Admin user already exists');
      return;
    }

    const passwordHash = bcrypt.hashSync('Admin123!', 12);

    await this.userModel.create({
      email: adminEmail,
      passwordHash,
      role: UserRole.ADMIN,
      status: UserStatus.ACTIVE,
      profile: {
        firstName: 'Delight',
        lastName: 'Admin',
      },
    });

    console.log('[SEED] Admin user created successfully');
  }

  private async seedDemoUser(): Promise<void> {
    const demoEmail = 'demo@delight.health';
    const existingDemo = await this.userModel.findOne({ email: demoEmail });

    if (existingDemo) {
      console.log('[SEED] Demo user already exists');
      return;
    }

    const passwordHash = bcrypt.hashSync('Demo1234!', 12);

    await this.userModel.create({
      email: demoEmail,
      passwordHash,
      role: UserRole.PATIENT,
      status: UserStatus.ACTIVE,
      profile: {
        firstName: 'Demo',
        lastName: 'User',
        language: 'en',
        timezone: 'UTC',
      },
      healthBaseline: {
        diabetesType: 'Type 2',
        allergies: [],
        chronicConditions: [],
        currentMedications: ['Metformin 500mg'],
        hba1c: 7.2,
        heightCm: 175,
        weightKg: 82,
      },
      personalityAssessment: {
        completed: false,
      },
      consent: {
        dataProcessing: true,
        healthDataSharing: true,
        marketingEmails: false,
        consentedAt: new Date(),
      },
      notificationPreferences: {
        push: true,
        email: true,
        sms: false,
        glucoseAlerts: true,
        mealReminders: true,
        medicationReminders: true,
      },
    });

    console.log('[SEED] Demo user created: demo@delight.health / Demo1234!');
  }
}
