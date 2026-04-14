import {
  Injectable,
  ConflictException,
  UnauthorizedException,
  BadRequestException,
} from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import { HttpService } from '@nestjs/axios';
import * as bcrypt from 'bcrypt';
import { User, UserDocument, UserRole, UserStatus } from './schemas/user.schema';
import { RegisterDto } from './dto/register.dto';
import { LoginDto } from './dto/login.dto';
import { TokenResponseDto } from './dto/token-response.dto';
import { firstValueFrom } from 'rxjs';
import { catchError, timeout } from 'rxjs/operators';

@Injectable()
export class AuthService {
  private aiServiceUrl: string;

  constructor(
    @InjectModel(User.name) private userModel: Model<UserDocument>,
    private jwtService: JwtService,
    private configService: ConfigService,
    private httpService: HttpService,
  ) {
    this.aiServiceUrl = this.configService.get<string>('AI_SERVICE_URL') || 'http://localhost:8000';
  }

  /**
   * Sync user to AI Service after registration
   * This ensures the AI service knows about the user
   */
  private async syncUserToAiService(userId: string, email: string, role: string): Promise<void> {
    try {
      // Create a system token for internal communication (uses same JWT_SECRET)
      const systemToken = this.jwtService.sign({
        sub: userId,
        email,
        role,
        type: 'system',
      });

      await firstValueFrom(
        this.httpService
          .post(
            `${this.aiServiceUrl}/api/v1/users/sync`,
            {
              user_id: userId,
              email,
              role,
            },
            {
              headers: { Authorization: `Bearer ${systemToken}` },
            },
          )
          .pipe(
            timeout(5000),
            catchError((err) => {
              // Log but don't fail registration if AI service is down
              console.warn('[AuthService] Failed to sync user to AI service:', err.message);
              return [];
            }),
          ),
      );
    } catch (error) {
      // Silently fail - user can still use the app, AI service will create user on first chat
      console.warn('[AuthService] AI service sync failed:', error);
    }
  }

  async register(dto: RegisterDto): Promise<TokenResponseDto> {
    const existingUser = await this.userModel.findOne({ email: dto.email });
    if (existingUser) {
      throw new ConflictException('Email already registered');
    }

    const passwordHash = await bcrypt.hash(dto.password, 12);

    const user = await this.userModel.create({
      email: dto.email,
      passwordHash,
      role: dto.role || UserRole.PATIENT,
      status: UserStatus.ACTIVE,
      profile: {
        firstName: dto.firstName,
        lastName: dto.lastName,
        language: dto.language || 'en',
      },
    });

    const payload = { sub: user._id.toString(), email: user.email, role: user.role };
    const accessToken = this.jwtService.sign(payload);

    // Sync user to AI Service (non-blocking, fails silently)
    this.syncUserToAiService(user._id.toString(), user.email, user.role).catch(() => {});

    return {
      accessToken,
      user: {
        id: user._id.toString(),
        email: user.email,
        role: user.role,
        status: user.status,
        profile: user.profile,
      },
    };
  }

  async login(dto: LoginDto): Promise<TokenResponseDto> {
    const user = await this.userModel.findOne({ email: dto.email });
    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const isPasswordValid = await bcrypt.compare(dto.password, user.passwordHash);
    if (!isPasswordValid) {
      throw new UnauthorizedException('Invalid credentials');
    }

    // Update status to ACTIVE if it was PENDING
    if (user.status === UserStatus.PENDING) {
      user.status = UserStatus.ACTIVE;
      await user.save();
    }

    const payload = { sub: user._id.toString(), email: user.email, role: user.role };
    const accessToken = this.jwtService.sign(payload);

    return {
      accessToken,
      user: {
        id: user._id.toString(),
        email: user.email,
        role: user.role,
        status: user.status,
        profile: user.profile,
      },
    };
  }

  async refresh(token: string): Promise<TokenResponseDto> {
    try {
      const payload = this.jwtService.verify(token);
      const user = await this.userModel.findById(payload.sub);

      if (!user) {
        throw new UnauthorizedException('User not found');
      }

      const newPayload = { sub: user._id.toString(), email: user.email, role: user.role };
      const newAccessToken = this.jwtService.sign(newPayload);

      return {
        accessToken: newAccessToken,
        user: {
          id: user._id.toString(),
          email: user.email,
          role: user.role,
          status: user.status,
        },
      };
    } catch (error) {
      throw new UnauthorizedException('Invalid or expired token');
    }
  }

  async logout(userId: string): Promise<{ success: boolean }> {
    return { success: true };
  }
}
