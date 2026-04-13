import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { User, UserDocument, UserStatus } from '../auth/schemas/user.schema';
import {
  UpdateProfileDto,
  HealthBaselineDto,
  PersonalityAssessmentDto,
  UpdateConsentDto,
  UpdateNotificationPreferencesDto,
} from './dto/user.dto';

@Injectable()
export class UsersService {
  constructor(
    @InjectModel(User.name) private userModel: Model<UserDocument>,
  ) {}

  async findById(userId: string): Promise<UserDocument> {
    const user = await this.userModel.findById(userId);
    if (!user) {
      throw new NotFoundException('User not found');
    }
    return user;
  }

  async findByEmail(email: string): Promise<UserDocument | null> {
    return this.userModel.findOne({ email });
  }

  async getCurrentUserProfile(userId: string): Promise<any> {
    const user = await this.findById(userId);
    const profile = user.toObject();
    delete profile.passwordHash;
    return profile;
  }

  async updateProfile(userId: string, dto: UpdateProfileDto): Promise<any> {
    const user = await this.findById(userId);

    user.profile = {
      ...user.profile,
      ...dto,
    };
    user.updatedAt = new Date();

    await user.save();

    const updatedProfile = user.toObject();
    delete updatedProfile.passwordHash;
    return updatedProfile;
  }

  async saveHealthBaseline(userId: string, dto: HealthBaselineDto): Promise<any> {
    const user = await this.findById(userId);

    user.healthBaseline = {
      ...user.healthBaseline,
      ...dto,
    };
    user.status = UserStatus.ACTIVE;
    user.updatedAt = new Date();

    await user.save();

    const result = user.toObject();
    delete result.passwordHash;
    return result;
  }

  async savePersonalityAssessment(
    userId: string,
    dto: PersonalityAssessmentDto,
  ): Promise<any> {
    const user = await this.findById(userId);

    user.personalityAssessment = {
      completed: true,
      results: dto.results,
      completedAt: new Date(),
    };
    user.updatedAt = new Date();

    await user.save();

    const result = user.toObject();
    delete result.passwordHash;
    return result;
  }

  async updateConsent(userId: string, dto: UpdateConsentDto): Promise<any> {
    const user = await this.findById(userId);

    user.consent = {
      ...user.consent,
      ...dto,
      consentedAt: new Date(),
    };
    user.updatedAt = new Date();

    await user.save();

    const result = user.toObject();
    delete result.passwordHash;
    return result;
  }

  async updateNotificationPreferences(
    userId: string,
    dto: UpdateNotificationPreferencesDto,
  ): Promise<any> {
    const user = await this.findById(userId);

    user.notificationPreferences = {
      ...user.notificationPreferences,
      ...dto,
    };
    user.updatedAt = new Date();

    await user.save();

    const result = user.toObject();
    delete result.passwordHash;
    return result;
  }
}
