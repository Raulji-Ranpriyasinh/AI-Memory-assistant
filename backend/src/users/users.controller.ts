import {
  Controller,
  Get,
  Put,
  Post,
  Body,
  UseGuards,
} from '@nestjs/common';
import { UsersService } from './users.service';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import {
  UpdateProfileDto,
  HealthBaselineDto,
  PersonalityAssessmentDto,
  UpdateConsentDto,
  UpdateNotificationPreferencesDto,
} from './dto/user.dto';

@Controller('users')
@UseGuards(JwtAuthGuard)
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @Get('me')
  async getCurrentUserProfile(@CurrentUser() user: any) {
    return this.usersService.getCurrentUserProfile(user.userId);
  }

  @Put('me')
  async updateProfile(
    @CurrentUser() user: any,
    @Body() dto: UpdateProfileDto,
  ) {
    return this.usersService.updateProfile(user.userId, dto);
  }

  @Post('me/onboarding')
  async saveHealthBaseline(
    @CurrentUser() user: any,
    @Body() dto: HealthBaselineDto,
  ) {
    return this.usersService.saveHealthBaseline(user.userId, dto);
  }

  @Post('me/personality-assessment')
  async savePersonalityAssessment(
    @CurrentUser() user: any,
    @Body() dto: PersonalityAssessmentDto,
  ) {
    return this.usersService.savePersonalityAssessment(user.userId, dto);
  }

  @Put('me/consent')
  async updateConsent(
    @CurrentUser() user: any,
    @Body() dto: UpdateConsentDto,
  ) {
    return this.usersService.updateConsent(user.userId, dto);
  }

  @Put('me/notification-preferences')
  async updateNotificationPreferences(
    @CurrentUser() user: any,
    @Body() dto: UpdateNotificationPreferencesDto,
  ) {
    return this.usersService.updateNotificationPreferences(user.userId, dto);
  }
}
