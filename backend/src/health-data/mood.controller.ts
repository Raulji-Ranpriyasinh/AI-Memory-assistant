import {
  Controller,
  Post,
  Get,
  Body,
  Query,
  UseGuards,
  Headers,
} from '@nestjs/common';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { HealthDataService } from './health-data.service';
import { AiProxyService } from '../ai-proxy/ai-proxy.service';
import { MoodEntryDto, MoodHistoryQueryDto } from './dto/mood.dto';

@Controller('mood')
@UseGuards(JwtAuthGuard)
export class MoodController {
  constructor(
    private readonly healthDataService: HealthDataService,
    private readonly aiProxyService: AiProxyService,
  ) {}

  @Post()
  async logMood(
    @Body() dto: MoodEntryDto,
    @CurrentUser() user: any,
    @Headers('authorization') authHeader: string,
  ) {
    const token = authHeader?.replace('Bearer ', '');
    const saved = await this.healthDataService.saveMoodEntry(user.userId, dto);

    let aiHint = null;
    try {
      const aiResponse = await this.aiProxyService.logMood(user.userId, dto, token);
      aiHint = aiResponse?.hint || aiResponse?.message || null;
    } catch {
      // AI service unavailable - continue without AI hint
    }

    return {
      success: true,
      data: {
        saved: true,
        aiHint,
      },
    };
  }

  @Get('history')
  async getHistory(
    @Query() query: MoodHistoryQueryDto,
    @CurrentUser() user: any,
  ) {
    const history = await this.healthDataService.getMoodHistory(
      user.userId,
      query.from ? new Date(query.from) : undefined,
      query.to ? new Date(query.to) : undefined,
      query.limit,
    );

    return {
      success: true,
      data: history,
    };
  }
}
