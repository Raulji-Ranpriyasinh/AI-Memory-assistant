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
import { BulkCgmReadingsDto, CgmSummaryQueryDto } from './dto/cgm.dto';

@Controller('cgm')
@UseGuards(JwtAuthGuard)
export class CgmController {
  constructor(
    private readonly healthDataService: HealthDataService,
    private readonly aiProxyService: AiProxyService,
  ) {}

  @Post('readings')
  async saveReadings(
    @Body() dto: BulkCgmReadingsDto,
    @CurrentUser() user: any,
    @Headers('authorization') authHeader: string,
  ) {
    const token = authHeader?.replace('Bearer ', '');
    const saved = await this.healthDataService.saveCgmReadings(user.userId, dto.readings);

    let aiResponse = null;
    try {
      aiResponse = await this.aiProxyService.processCgmReadings(
        user.userId,
        dto.readings,
        token,
      );
    } catch {
      // AI service unavailable - continue without AI response
    }

    return {
      success: true,
      data: {
        saved: saved.length,
        aiResponse,
      },
    };
  }

  @Get('summary')
  async getSummary(
    @Query() query: CgmSummaryQueryDto,
    @CurrentUser() user: any,
  ) {
    const summary = await this.healthDataService.getCgmSummary(
      user.userId,
      query.period,
      query.from ? new Date(query.from) : undefined,
      query.to ? new Date(query.to) : undefined,
    );

    return {
      success: true,
      data: summary,
    };
  }

  @Get('history')
  async getHistory(
    @Query() query: CgmSummaryQueryDto,
    @CurrentUser() user: any,
  ) {
    const history = await this.healthDataService.getCgmHistory(
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
